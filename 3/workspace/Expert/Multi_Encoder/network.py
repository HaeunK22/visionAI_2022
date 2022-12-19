import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np

from basis import ResBlk, conv1x1, conv3x3

import torch.nn as nn
import torch


def conv2d(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True))


def conv(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))


class VGG19(nn.Module):
	def __init__(self):
		super(VGG19, self).__init__()
		#self.mean = torch.tensor([0.485, 0.456, 0.406]).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(device)
		#self.std = torch.tensor([0.229, 0.224, 0.225]).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(device)
		self.conv1_1 = nn.Sequential(conv(3, 64), nn.ReLU())
		self.conv1_2 = nn.Sequential(conv(64, 64), nn.ReLU())
		self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)
		self.conv2_1 = nn.Sequential(conv(64, 128), nn.ReLU())
		self.conv2_2 = nn.Sequential(conv(128, 128), nn.ReLU())
		self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)
		self.conv3_1 = nn.Sequential(conv(128, 256), nn.ReLU())
		self.conv3_2 = nn.Sequential(conv(256, 256), nn.ReLU())
		self.conv3_3 = nn.Sequential(conv(256, 256), nn.ReLU())
		self.conv3_4 = nn.Sequential(conv(256, 256), nn.ReLU())
		self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)
		self.conv4_1 = nn.Sequential(conv(256, 512), nn.ReLU())
		self.conv4_2 = nn.Sequential(conv(512, 512), nn.ReLU())


	def load_model(self, model_file):
		vgg19_dict = self.state_dict()
		pretrained_dict = torch.load(model_file)

		vgg19_keys = vgg19_dict.keys()
		pretrained_keys = pretrained_dict.keys()
		for k, pk in zip(vgg19_keys, pretrained_keys):
			vgg19_dict[k] = pretrained_dict[pk]
		self.load_state_dict(vgg19_dict)

	def forward(self, input_images):
		#input_images = (input_images - self.mean) / self.std
		feature = {}
		feature['conv1_1'] = self.conv1_1(input_images)
		feature['conv1_2'] = self.conv1_2(feature['conv1_1'])
		feature['pool1'] = self.pool1(feature['conv1_2'])
		feature['conv2_1'] = self.conv2_1(feature['pool1'])
		feature['conv2_2'] = self.conv2_2(feature['conv2_1'])
		feature['pool2'] = self.pool2(feature['conv2_2'])
		feature['conv3_1'] = self.conv3_1(feature['pool2'])
		feature['conv3_2'] = self.conv3_2(feature['conv3_1'])
		feature['conv3_3'] = self.conv3_3(feature['conv3_2'])
		feature['conv3_4'] = self.conv3_4(feature['conv3_3'])
		feature['pool3'] = self.pool3(feature['conv3_4'])
		feature['conv4_1'] = self.conv4_1(feature['pool3'])
		x = self.conv4_2(feature['conv4_1'])


		return x

class StyleEncoder(nn.Module):
	def __init__(self, dim):
		super(StyleEncoder, self).__init__()
		self.layers = [4, 4, 4, 4]
		self.planes = [64, 128, 256, 512]

		self.num_layers = sum(self.layers)
		self.inplanes = self.planes[0]
		self.conv0 = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1, bias=False)
		self.conv1 = nn.Conv2d(3, self.planes[0], kernel_size=7, stride=2, padding=3, bias=False)
		self.bias1 = nn.Parameter(torch.zeros(1))
		self.actv = nn.PReLU()
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.layer1 = self._make_layer(ResBlk, self.planes[0], self.layers[0])
		self.layer2 = self._make_layer(ResBlk, self.planes[1], self.layers[1], stride=2)
		self.layer3 = self._make_layer(ResBlk, self.planes[2], self.layers[2], stride=2)
		self.layer4 = self._make_layer(ResBlk, self.planes[3], self.layers[3], stride=2)

		self.q_conv = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=False)
		self.k_conv = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=False)
		self.v_conv = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=False)
		self.sfm = nn.Softmax(dim=1)
		self.gap = nn.AdaptiveAvgPool2d(1)
		self.gmp = nn.AdaptiveMaxPool2d(1)
		self.bias2 = nn.Parameter(torch.zeros(1))

		self.fc = nn.Linear(self.planes[3], dim)

		self._reset_params()

	def _reset_params(self):
		for m in self.modules():
			if isinstance(m, ResBlk):
				nn.init.normal_(m.conv1.weight, mean=0, std=np.sqrt(2 / (m.conv1.weight.shape[0] * np.prod(m.conv1.weight.shape[2:]))) * self.num_layers ** (-0.5))
				nn.init.constant_(m.conv2.weight, 0)
				if m.downsample is not None:
					nn.init.normal_(m.downsample.weight, mean=0, std=np.sqrt(2 / (m.downsample.weight.shape[0] * np.prod(m.downsample.weight.shape[2:]))))
			elif isinstance(m, nn.Linear):
				nn.init.xavier_normal_(m.weight)
				nn.init.constant_(m.bias, 0)

	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes:
			downsample = conv1x1(self.inplanes, planes, stride)

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes
		for _ in range(1, blocks):
			layers.append(block(self.inplanes, planes))

		return nn.Sequential(*layers)

	def forward(self, x):
		x = self.conv0(x)
		x = self.conv1(x)
		x = self.actv(x + self.bias1)
		x = self.maxpool(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		b, c, h, w = x.shape
		q = self.q_conv(x).view(b, c, -1)
		k = self.k_conv(x).view(b, c, -1)
		v = self.v_conv(x).view(b, c, -1)

		q = q.transpose(1,2)
		qk = q.matmul(k)
		qk = self.sfm(qk)
		x = v.matmul(qk)
		x = x.view(b, c, h, w)

		avg_x = self.gap(x)
		max_x = self.gmp(x)

		x = (max_x + avg_x).flatten(1)
		x = self.fc(x + self.bias2)

		x = F.normalize(x, p=2, dim=1)

		return x


class Proxy(nn.Module):
	def __init__(self, dim, cN):
		super(Proxy, self).__init__()
		self.fc = Parameter(torch.Tensor(dim, cN))
		torch.nn.init.xavier_normal_(self.fc)
		self.softmax = nn.Softmax(dim=1)
	def forward(self, input):
		centers = F.normalize(self.fc, p=2, dim=0)
		simInd = input.matmul(centers)
		simInd = self.softmax(simInd)
		return simInd

class CProxy(nn.Module):
	def __init__(self, dim, cN):
		super(CProxy, self).__init__()

		self.fc = Parameter(torch.Tensor(dim, cN))
		torch.nn.init.xavier_normal_(self.fc)
		self.softmax = nn.Softmax(dim=1)
	def forward(self, input):
		centers = F.normalize(self.fc, p=2, dim=0)
		simInd = input.matmul(centers)
		simInd = self.softmax(simInd)
		return simInd

class Stylish(nn.Module):
	def __init__(self, dim, obj):
		super(Stylish, self).__init__()
		self.encoder = StyleEncoder(dim)
		self.cproxy = CProxy(dim, obj)
	def forward(self, x):
		x = self.encoder(F.adaptive_avg_pool2d(x, (224, 224)))
		out2 = self.cproxy(x)

		return x, out2

class Style(nn.Module):
	def __init__(self, dim, obj):
		super(Style, self).__init__()
		self.encoder = VGG19()
		self.encoder.load_model('/nasspace/vgg19-dcbb9e9d.pth')
		self.conv1 = nn.Conv2d(1, 3, 3, 1, 1)
		self.gap = nn.AdaptiveAvgPool2d(1)
		self.gmp = nn.AdaptiveMaxPool2d(1)
		self.bias2 = nn.Parameter(torch.zeros(1))
		self.fc = nn.Linear(512, 512)
		self.cproxy = CProxy(dim, obj)
	def forward(self, x):
		x = self.encoder(self.conv1(F.adaptive_avg_pool2d(x, (224, 224))))
		avg_x = self.gap(x)
		max_x = self.gmp(x)

		x = (max_x + avg_x).flatten(1)
		x = self.fc(x + self.bias2)

		x = F.normalize(x, p=2, dim=1)

		out2 = self.cproxy(x)

		return x, out2

class Mixer(nn.Module):
	def __init__(self, dim):
		super(Mixer, self).__init__()
		self.fc1 = Parameter(torch.Tensor(dim*2, dim))
		self.fc2 = Parameter(torch.Tensor(dim, dim))
		self.fc3 = Parameter(torch.Tensor(dim, dim))
		torch.nn.init.xavier_normal_(self.fc1)
		torch.nn.init.xavier_normal_(self.fc2)
		torch.nn.init.xavier_normal_(self.fc3)
		self.softmax = nn.Softmax(dim=1)
	def forward(self, x):
		x = x.matmul(self.fc1)
		x = x.matmul(self.fc2)
		x = x.matmul(self.fc3)
		x = F.normalize(x, p=2, dim=1)
		return x

class Encoder(nn.Module):
	def __init__(self, dim, style, obj):
		super(Encoder, self).__init__()
		self.style_head = Style(dim, style)
		self.object_head = Stylish(dim, obj)
		self.mixer = Mixer(dim)

	def forward(self, x):
		style_proxy, style_id = self.style_head(x)
		obj_proxy, obj_id = self.object_head(x)
		x = torch.cat([style_proxy, obj_proxy],dim=1)
		x = self.mixer(x)
		return x, style_id, obj_id


