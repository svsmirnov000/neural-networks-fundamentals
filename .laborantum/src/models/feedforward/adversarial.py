import torch
import copy


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, signal, strength):
        ctx.strength = strength
        return signal.view_as(signal)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.strength * grad_output, None


class GradientReversalLayer(torch.nn.Module):
    def __init__(self, strength=1.0):
        super().__init__()
        self.strength = float(strength)

    def forward(self, signal):
        return GradientReversalFunction.apply(signal, self.strength)


class GAN(torch.nn.Module):
    def __init__(
            self,
            channels,
            gradient_reversal_strength=1.0,
            activation=lambda: torch.nn.LeakyReLU(negative_slope=0.5)
        ):
        super().__init__()
        
        self.generator_discriminator_bridge = GradientReversalLayer(gradient_reversal_strength)
        self.gradient_reversal = self.generator_discriminator_bridge
        
        self.generator = torch.nn.Sequential(
            torch.nn.Linear(channels[0], channels[1]),
            activation(),
            torch.nn.Linear(channels[1], channels[2]),
            torch.nn.Tanh(),
        )
        
        self.discriminator = torch.nn.Sequential(
            torch.nn.Linear(channels[2], channels[1]),
            activation(),
            torch.nn.Linear(channels[1], channels[0]),
        )

        self.classifier = torch.nn.Linear(channels[0], 1)

    def discriminate(self, signal):
        signal = signal.reshape(signal.shape[0], -1)
        features = self.discriminator(signal)
        return self.classifier(features).flatten()

    def forward(self, batch):
        generated = self.generator(batch['data']['noise'])
        fake_input = self.generator_discriminator_bridge(generated)
        if 'real' in batch['data'] and batch['data']['real'] is not None:
            real_input = batch['data']['real']
        elif 'image' in batch['data'] and batch['data']['image'] is not None:
            real_input = batch['data']['image']
        else:
            real_input = None
        if real_input is not None:
            real_input = real_input.flatten(start_dim=1)

        disc_input = [fake_input]
        if real_input is not None:
            disc_input.append(real_input)
        disc_input = torch.cat(disc_input, dim=0)

        logits = self.discriminate(disc_input)
        
        bs = generated.shape[0]
        fake_logits = logits[:bs]
        real_logits = logits[bs:] if real_input is not None else None
        
        batch['signals'] = {
            'generated': generated,
            'discriminator_logits': logits,
            'fake_logits': fake_logits,
            'discriminator_scores': logits,
            'fake_scores': fake_logits,
        }

        if real_logits is not None:
            batch['signals']['real_logits'] = real_logits
            batch['signals']['real_scores'] = real_logits

        batch['postprocessed'] = {
            'discriminator_score': logits,
            'discriminator_probability': torch.sigmoid(logits),
            'fake_score': fake_logits,
            'fake_probability': torch.sigmoid(fake_logits),
        }

        if real_logits is not None:
            batch['postprocessed']['real_score'] = real_logits
            batch['postprocessed']['real_probability'] = torch.sigmoid(real_logits)

        return batch