import os
import json
from enum import Enum
import torch
from torch import nn
from tensorboardX import SummaryWriter
from torch_tools.modules import DataParallelPassthrough

from utils import make_noise, is_conditional
from train_log import MeanTracker
from visualization import make_interpolation_chart, fig_to_image
from latent_deformator import DeformatorType
import numpy as np
import random
from classifier import CRDiscriminator,Classifier


class ShiftDistribution(Enum):
    NORMAL = 0,
    UNIFORM = 1,


class Params(object):
    def __init__(self, **kwargs):
        self.shift_scale = 6.0
        self.min_shift = 0.5
        self.shift_distribution = ShiftDistribution.UNIFORM

        self.deformator_lr = 0.0001
        self.shift_predictor_lr = 0.0001
        self.n_steps = int(1e+5)
        self.batch_size = 32

        self.directions_count = None
        self.max_latent_dim = None

        self.label_weight = 1.0
        self.shift_weight = 0.25

        self.steps_per_log = 10
        self.steps_per_img_log = 1000
        self.steps_per_backup = 1000

        self.truncation = None

        for key, val in kwargs.items():
            if val is not None:
                self.__dict__[key] = val


class Trainer(object):
    def __init__(self, params=Params(), out_dir='', verbose=True):
        if verbose:
            print('Trainer inited with:\n{}'.format(str(params.__dict__)))
        self.p = params
        self.log_dir = os.path.join(out_dir, 'logs')
        os.makedirs(self.log_dir, exist_ok=True)
        self.cross_entropy = nn.CrossEntropyLoss()

        tb_dir = os.path.join(out_dir, 'tensorboard')
        self.models_dir = os.path.join(out_dir, 'models')
        self.images_dir = os.path.join(self.log_dir, 'images')
        os.makedirs(tb_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)

        self.checkpoint = os.path.join(out_dir, 'checkpoint_70000.pt')
        self.writer = SummaryWriter(tb_dir)
        self.out_json = os.path.join(self.log_dir, 'stat.json')
        self.fixed_test_noise = None
        self.out_dir = out_dir

    def make_shifts(self, latent_dim):
        target_indices = torch.randint(
            0, self.p.directions_count, [self.p.batch_size]).cuda()
        if self.p.shift_distribution == ShiftDistribution.NORMAL:
            shifts = torch.randn(target_indices.shape).cuda()
        elif self.p.shift_distribution == ShiftDistribution.UNIFORM:
            shifts = 2.0 * torch.rand(target_indices.shape).cuda() - 1.0

        shifts = self.p.shift_scale * shifts
        shifts[(shifts < self.p.min_shift) & (shifts > 0)] = self.p.min_shift
        shifts[(shifts > -self.p.min_shift) & (shifts < 0)] = -self.p.min_shift

        try:
            latent_dim[0]
            latent_dim = list(latent_dim)
        except Exception:
            latent_dim = [latent_dim]

        z_shift = torch.zeros([self.p.batch_size] + latent_dim).cuda()
        for i, (index, val) in enumerate(zip(target_indices, shifts)):
            z_shift[i][index] += val

        return target_indices, shifts, z_shift

    def log_train(self, step, should_print=True, stats=()):
        if should_print:
            out_text = '{}% [step {}]'.format(int(100 * step / self.p.n_steps), step)
            for named_value in stats:
                out_text += (' | {}: {:.2f}'.format(*named_value))
            print(out_text)
        for named_value in stats:
            self.writer.add_scalar(named_value[0], named_value[1], step)

        with open(self.out_json, 'a+') as out:
            out.write('step :' + str(step))
            stat_dict = {named_value[0]: named_value[1] for named_value in stats}
            json.dump(stat_dict, out)
            out.write('\n')

    def log_interpolation(self, G, deformator, step):
        noise = make_noise(1, G.dim_z, self.p.truncation).cuda()
        if self.fixed_test_noise is None:
            self.fixed_test_noise = noise.clone()
        for z, prefix in zip([noise, self.fixed_test_noise], ['rand', 'fixed']):
            fig = make_interpolation_chart(
                G, deformator, z=z, shifts_r=3 * self.p.shift_scale, shifts_count=3, dims_count=15,
                dpi=500)

            self.writer.add_figure('{}_deformed_interpolation'.format(prefix), fig, step)
            fig_to_image(fig).convert("RGB").save(
                os.path.join(self.images_dir, '{}_{}.jpg'.format(prefix, step)))


    def save_checkpoint(self, deformator, shift_predictor, deformator_opt,shift_predictor_opt,step):
        state_dict = {
            'step': step,
            'rng_state': torch.get_rng_state(),
            'np_rng_state': np.random.get_state(),
            'random_rng_state': random.getstate(),
            'deformator': deformator.state_dict(),
            'shift_predictor': shift_predictor.state_dict(),
            'deformator_opt': deformator_opt.state_dict(),
            'shift_predictor_opt':shift_predictor_opt.state_dict(),
        }
        torch.save(state_dict, self.checkpoint)
        torch.save(state_dict, os.path.join(self.out_dir, 'checkpoint_'+ str(step)+'.pt'))


    def log_accuracy(self, G, deformator, shift_predictor, step):
        deformator.eval()
        shift_predictor.eval()

        accuracy = validate_classifier(G, deformator, shift_predictor, trainer=self)
        self.writer.add_scalar('accuracy', accuracy.item(), step)

        deformator.train()
        shift_predictor.train()
        return accuracy

    def log(self, G, deformator, shift_predictor,shift_predictor_opt,deformator_opt, step, avgs):
        if step % self.p.steps_per_log == 0:
            self.log_train(step, True, [avg.flush() for avg in avgs])

        if step % self.p.steps_per_img_log == 0:
            self.log_interpolation(G, deformator, step)

        if step % self.p.steps_per_backup == 0 and step > 0:
            accuracy = self.log_accuracy(G, deformator, shift_predictor, step)
            print('Step {} accuracy: {:.3}'.format(step, accuracy.item()))
            self.save_checkpoint(deformator, shift_predictor,deformator_opt,shift_predictor_opt,step)


    def train(self, G, deformator, shift_predictor,seed , resume_train, multi_gpu=False):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        step = 1

        if resume_train:
            checkpoint = torch.load(self.checkpoint)
            step = checkpoint['step']
            step = step + 1
            torch.set_rng_state(checkpoint['rng_state'])
            np.random.set_state(checkpoint['np_rng_state'])
            random.setstate(checkpoint['random_rng_state'])

        G.cuda().eval()
        deformator.cuda().train()
        shift_predictor.cuda().train()

        should_gen_classes = is_conditional(G)
        if multi_gpu:
            G = DataParallelPassthrough(G)

        if resume_train:
            deformator.load_state_dict(checkpoint['deformator'])
            shift_predictor.load_state_dict(checkpoint['shift_predictor'])

        deformator_opt = torch.optim.Adam(deformator.parameters(), lr=self.p.deformator_lr) \
            if deformator.type not in [DeformatorType.ID, DeformatorType.RANDOM] else None
        shift_predictor_opt = torch.optim.Adam(
            shift_predictor.parameters(), lr=self.p.shift_predictor_lr)

        if resume_train:
            deformator_opt.load_state_dict(checkpoint['deformator_opt'])
            shift_predictor_opt.load_state_dict(checkpoint['shift_predictor_opt'])

        avgs = MeanTracker('percent'), MeanTracker('loss'), MeanTracker('direction_loss'),\
               MeanTracker('shift_loss')
        avg_correct_percent, avg_loss, avg_label_loss, avg_shift_loss = avgs
        classifier = CRDiscriminator(dim_c_cont=2).cuda()
        cr_optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))

        adverserial_loss = torch.nn.CrossEntropyLoss()
        training_loss = []

        for i in range(5000):
            # G.zero_grad()
            # deformator.zero_grad()
            # shift_predictor.zero_grad()

            cr_optimizer.zero_grad()

            label_real = torch.full((128,), 1, dtype=torch.long, device='cuda')
            label_fake = torch.full((128,), 0, dtype=torch.long, device='cuda')
            labels = torch.cat((label_real, label_fake))

            z = make_noise(self.p.batch_size, G.dim_z, self.p.truncation).cuda()
            target_indices, shifts, basis_shift = self.make_shifts(deformator.input_dim)

            if should_gen_classes:
                classes = G.mixed_classes(z.shape[0])

            # Deformation
            shift = deformator(basis_shift)
            if should_gen_classes:
                imgs = G(z, classes)
                imgs_shifted = G.gen_shifted(z, shift, classes)
            else:
                imgs = G(z)
                imgs_disentagled = G.gen_shifted(z, shift)

            entangled_vectors = torch.zeros((self.p.batch_size,self.p.directions_count))
            for vec in entangled_vectors:
                num_ones = random.randint(2,10)
                one_idx = random.sample(range(self.p.directions_count),k=num_ones)
                for idx in one_idx:
                    vec[idx] = 1

            shifts_new = torch.randn(entangled_vectors.shape).cuda()
            shifted = entangled_vectors.cuda()*shifts_new
            shifted = self.p.shift_scale * shifted

            shifted[(shifted < self.p.min_shift) & (shifted > 0)] = self.p.min_shift
            shifted[(shifted > -self.p.min_shift) & (shifted < 0)] = -self.p.min_shift

            shift_entangled = deformator(shifted)
            imgs_entangled = G.gen_shifted(z, shift_entangled)

            images = torch.cat((imgs_disentagled.detach(), imgs_entangled.detach()))
            ref_images = torch.cat((imgs.detach(), imgs.detach()))

            shuffled_indices = torch.randint(0,images.size(0), (images.size(0),))
            ref_images = ref_images[shuffled_indices]
            images = images[shuffled_indices]
            labels = labels[shuffled_indices]

            prob = classifier(ref_images.cuda(), images.cuda())
            loss_dis = adverserial_loss(prob, labels)
            loss_dis.backward()
            cr_optimizer.step()
            training_loss.append(loss_dis.item())

            if i % 100 == 0 and i != 0:
                correct = 0
                total = 0
                classifier.eval()
                with torch.no_grad():
                    for k in range(200):
                        label_real = torch.full((self.p.batch_size,), 1, dtype=torch.float32, device='cuda')
                        label_fake = torch.full((self.p.batch_size,), 0, dtype=torch.float32, device='cuda')
                        z = make_noise(self.p.batch_size, G.dim_z, self.p.truncation).cuda()
                        target_indices, shifts, basis_shift = self.make_shifts(deformator.input_dim)
                        shift = deformator(basis_shift)
                        ref_images = G(z)
                        imgs_disentangled = G.gen_shifted(z, shift)
                        entangled_vectors = torch.zeros((self.p.batch_size, self.p.directions_count))
                        for vec in entangled_vectors:
                            num_ones = random.randint(2, 10)
                            one_idx = random.sample(range(64), k=num_ones)
                            for idx in one_idx:
                                vec[idx] = 1

                        shifts_new = torch.randn(entangled_vectors.shape).cuda()
                        shifted = entangled_vectors.cuda() * shifts_new
                        shifted = self.p.shift_scale * shifted

                        shifted[(shifted < self.p.min_shift) & (shifted > 0)] = self.p.min_shift
                        shifted[(shifted > -self.p.min_shift) & (shifted < 0)] = -self.p.min_shift

                        shift_entangled = deformator(shifted)
                        imgs_entangled = G.gen_shifted(z, shift_entangled)

                        prob_disentangle = classifier(ref_images.cuda(), imgs_disentangled.cuda())
                        prob_entangle = classifier(ref_images.cuda(), imgs_entangled.cuda())
                        _, predicted_dis = torch.max(prob_disentangle, 1)
                        _, predicted_ent = torch.max(prob_entangle, 1)
                        predicted = torch.cat((predicted_dis, predicted_ent))
                        labels = torch.cat((label_real, label_fake))

                        # Total number of labels
                        total += labels.size(0)

                        # Total correct predictions
                        correct += (predicted.view(-1) == labels.view(-1)).sum()
                    classifier.train()
                    accuracy = 100 * correct.item() / total

                    print('training loss : ', sum(training_loss) / len(training_loss), "accuracy :", accuracy)
                    training_loss = []
        torch.save(classifier, 'pretrained_DISclassifier')






                    # logits, shift_prediction = shift_predictor(imgs, imgs_shifted)
            # logit_loss = self.p.label_weight * self.cross_entropy(logits, target_indices)
            # shift_loss = self.p.shift_weight * torch.mean(torch.abs(shift_prediction - shifts))
            #
            # # total loss
            # loss = logit_loss + shift_loss
            # loss.backward()
            #
            # if deformator_opt is not None:
            #     deformator_opt.step()
            # shift_predictor_opt.step()

            # update statistics trackers
            # avg_correct_percent.add(torch.mean(
            #         (torch.argmax(logits, dim=1) == target_indices).to(torch.float32)).detach())
            # avg_loss.add(loss.item())
            # avg_label_loss.add(logit_loss.item())
            # avg_shift_loss.add(shift_loss)
            #
            # self.log(G, deformator, shift_predictor,shift_predictor_opt,deformator_opt,i, avgs)


@torch.no_grad()
def validate_classifier(G, deformator, shift_predictor, params_dict=None, trainer=None):
    n_steps = 100
    if trainer is None:
        trainer = Trainer(params=Params(**params_dict), verbose=False)

    percents = torch.empty([n_steps])
    for step in range(n_steps):
        z = make_noise(trainer.p.batch_size, G.dim_z, trainer.p.truncation).cuda()
        target_indices, shifts, basis_shift = trainer.make_shifts(deformator.input_dim)

        imgs = G(z)
        imgs_shifted = G.gen_shifted(z, deformator(basis_shift))

        logits, _ = shift_predictor(imgs, imgs_shifted)
        percents[step] = (torch.argmax(logits, dim=1) == target_indices).to(torch.float32).mean()

    return percents.mean()
