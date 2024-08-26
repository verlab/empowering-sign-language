import torch
import torchvision.models
from facenet_pytorch import InceptionResnetV1


class IdentityEncoder(torch.nn.Module):
    def __init__(self, cfg = None):
        super(IdentityEncoder, self).__init__()
        self.cfg = cfg
        facenet = InceptionResnetV1(pretrained='vggface2').eval()
        self.facenet = torch.nn.Sequential(*list(facenet.children()))[:-5]

        #not really a deconv
        self.deconv1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(1792, 1024, kernel_size=(3, 3), stride = (1, 1), padding = 0),
            torch.nn.BatchNorm2d(1024),
            torch.nn.ELU(),            
        )

        self.deconv2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(1024, 512, kernel_size=(4, 4), stride = (4, 4), padding = 0),
            torch.nn.BatchNorm2d(512),
            torch.nn.ELU(),            
        )
    
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3),
            torch.nn.BatchNorm2d(512),
            torch.nn.ELU(),
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3),
            torch.nn.BatchNorm2d(512),
            torch.nn.ELU(),
            torch.nn.MaxPool2d(kernel_size=3)
        )

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 256, kernel_size=2),
            torch.nn.BatchNorm2d(256),
            torch.nn.ELU()
        )

        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, kernel_size=2, stride = 2),
            torch.nn.BatchNorm2d(256),
            torch.nn.ELU()
        )

        # Don't update params in VGG16
        for param in self.facenet.parameters():
            param.requires_grad = False

    def forward(self, image):
        
        features = self.facenet(image)
        features = self.deconv1(features)
        features = self.deconv2(features)
        features = self.layer1(features)
        features = self.layer2(features)
        features = self.layer3(features)
        features = self.layer4(features)
        
        return features