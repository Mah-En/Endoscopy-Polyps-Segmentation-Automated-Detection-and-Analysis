import sys
sys.path.append('../')
from pycore.tikzeng import *

# ResUNet (ResNet34 encoder, bilinear upsampling + 1x1 shrink + DoubleConv)
# NOTE: final head is drawn as 1 channel (binary). Change n_filer in "head" if you have >1 class.

arch = [
    to_head('..'),
    to_cor(),
    to_begin(),

    # =========================
    # INPUT IMAGE (PNG)
    # =========================
    # Renders the image as a front-facing plane. Adjust offset/size as desired.
    # Ensure 112.png is available relative to the .tex output.
    
    to_input('sample.jpg', to='(-2,0,0)', name='input', width=6, height=6),
    to_Conv("in_anchor", s_filer=512, n_filer=1, offset="(0,0,0)", to="(-1.2,0,0)", height=0.01, depth=0.01, width=0.01),


    # =========================
    # ENCODER (ResNet34 features)
    # =========================
    # layer0: conv1(7x7,s2)+bn+relu -> 1/2, 64ch
    to_Conv("enc0", s_filer=512, n_filer=64, offset="(0,0,0)", to="(0,0,0)", height=64, depth=64, width=2),

    # connect input image to first block
    # to_connection("input", "enc0"),
    to_connection("in_anchor", "enc0"),

    # (maxpool lives inside layer1; show an explicit down box to 1/4)
    to_Pool("down0", offset="(0,0,0)", to="(enc0-east)", height=48, depth=48, width=1),

    # layer1: -> 1/4, 64ch
    to_Conv("enc1", s_filer=256, n_filer=64, offset="(1,0,0)", to="(down0-east)", height=48, depth=48, width=2.2),
    to_connection("down0", "enc1"),

    # stride down to 1/8 (represented as pool for drawing)
    to_Pool("down1", offset="(0,0,0)", to="(enc1-east)", height=36, depth=36, width=1),

    # layer2: -> 1/8, 128ch
    to_Conv("enc2", s_filer=128, n_filer=128, offset="(0.8,0,0)", to="(down1-east)", height=36, depth=36, width=2.6),
    to_connection("down1", "enc2"),

    # stride down to 1/16
    to_Pool("down2", offset="(0,0,0)", to="(enc2-east)", height=24, depth=24, width=1),

    # layer3: -> 1/16, 256ch
    to_Conv("enc3", s_filer=64, n_filer=256, offset="(0.6,0,0)", to="(down2-east)", height=24, depth=24, width=3.2),
    to_connection("down2", "enc3"),

    # stride down to 1/32
    to_Pool("down3", offset="(0,0,0)", to="(enc3-east)", height=16, depth=16, width=1),

    # layer4: -> 1/32, 512ch
    to_Conv("enc4", s_filer=32, n_filer=512, offset="(0.6,0,0)", to="(down3-east)", height=16, depth=16, width=3.6),
    to_connection("down3", "enc4"),

    # =========================
    # DECODER (Up + 1x1 shrink + DoubleConv) with skips
    # =========================
    # up4: x4 (512) -> up -> 1x1 to 256 -> concat x3(256) -> DoubleConv 256 @ 1/16
    to_UnPool("up4", offset="(1.0,0,0)", to="(enc4-east)", height=24, depth=24, width=1),          # -> 1/16
    to_Conv("shrink4", s_filer=32, n_filer=256, offset="(0,0,0)", to="(up4-east)", height=24, depth=24, width=1.4),    
    to_Conv("dec4", s_filer=32, n_filer=256, to="(shrink4-east)", height=24, depth=24, width=2.8),


    # up3: -> 1/8, 128
    to_UnPool("up3", offset="(1.2,0,0)", to="(dec4-east)", height=36, depth=36, width=1),          # -> 1/8
    to_Conv("shrink3", s_filer=64, n_filer=128, offset="(0,0,0)", to="(up3-east)", height=36, depth=36, width=1.2),
    to_Conv("dec3", s_filer=64, n_filer=128, offset="(0,0,0)", to="(shrink3-east)", height=36, depth=36, width=2.4),

    # up2: -> 1/4, 64
    to_UnPool("up2", offset="(1.0,0,0)", to="(dec3-east)", height=48, depth=48, width=1),          # -> 1/4
    to_Conv("shrink2", s_filer=128, n_filer=64, offset="(0,0,0)", to="(up2-east)", height=48, depth=48, width=1.0),
    to_Conv("dec2", s_filer=128, n_filer=64, offset="(0,0,0)", to="(shrink2-east)", height=48, depth=48, width=2.0),

    # up1: -> 1/2, 64
    to_UnPool("up1", offset="(1.2,0,0)", to="(dec2-east)", height=64, depth=64, width=1),          # -> 1/2
    to_Conv("shrink1", s_filer=256, n_filer=64, offset="(0,0,0)", to="(up1-east)", height=64, depth=64, width=1.0),
    to_Conv("dec1", s_filer=256, n_filer=64, offset="(0,0,0)", to="(shrink1-east)", height=64, depth=64, width=2.0),

    # up0: final up to full res, DoubleConv(64,64)
    to_UnPool("up0", offset="(1.0,0,0)", to="(dec1-east)", height=80, depth=80, width=1),          # -> 1/1
    to_Conv("dec0", s_filer=512, n_filer=64, offset="(0,0,0)", to="(up0-east)", height=80, depth=80, width=2.0),

    # head: 1x1 conv -> n_classes (here 1; change if needed)
    to_Conv("head", s_filer=512, n_filer=1, offset="(0.8,0,0)", to="(dec0-east)", height=80, depth=80, width=0.8),

    # =========================
    # MAIN FLOW CONNECTIONS
    # =========================
    to_connection("enc0", "down0"),
    to_connection("enc1", "down1"),
    to_connection("enc2", "down2"),
    to_connection("enc3", "down3"),
    to_connection("enc4", "up4"),
    to_connection("dec4", "up3"),
    to_connection("dec3", "up2"),
    to_connection("dec2", "up1"),
    to_connection("dec1", "up0"),
    to_connection("dec0", "head"),

    # =========================
    # SKIP CONNECTIONS (encoder → decoder blocks)
    # =========================
    to_skip(of="enc3", to="dec4"),  # x3 -> dec4
    to_skip(of="enc2", to="dec3"),  # x2 -> dec3
    to_skip(of="enc1", to="dec2"),  # x1 -> dec2
    to_skip(of="enc0", to="dec1"),  # x0 -> dec1

    to_end()
]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex')

if __name__ == '__main__':
    main()
