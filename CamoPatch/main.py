from torchvision import transforms
from adversarial_score import FaceVerification
import numpy as np
import argparse
import os
from time import time
from get_architech import get_model
from torchvision import transforms
from PIL import *


def pytorch_switch(tensor_image):
    return tensor_image.permute(1, 2, 0)


from CamoPatch import Attack
if __name__ == "__main__":

    load_image = transforms.Compose([
        transforms.Resize((224,224)),
        # transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model", help="0 or 1", type=int, default=0)
    parser.add_argument("--model_name", type=str, help="resnet_vggface / resnet_webface / estnet_vggface_student", default="resnet_vggface")
    parser.add_argument("--N", help='Population size', type=int, default=10)
    parser.add_argument("--temp", type=float, default=300.)
    parser.add_argument("--mut", help="Mutation prob" ,type=float, default=0.1)
    parser.add_argument("--s", help="Patch size", type=int, default=16)
    parser.add_argument("--queries", help="Number of generations", type=int, default=10000)
    parser.add_argument("--li", help="Update location period", type=int, default=4)

    parser.add_argument("--image1_dir", type=str, help="Image1 File directory - image which we will attack")
    parser.add_argument("--image2_dir", type=str, help="Image2 File directory - image to compare")
    parser.add_argument("--true_label", type=int, help="the true label of 2 images")
    parser.add_argument("--save_directory", type=str, help="Where to store the results files", default="results/")
    parser.add_argument("--device", type=str, help="device", default="cuda")
    args = parser.parse_args()

    os.makedirs(args.save_directory, exist_ok=True)
    model = get_model(args.model_name, args.device)
    img1_dir = args.image1_dir
    img2_dir = args.image2_dir
    x_test1 = load_image(Image.open(img1_dir))
    x_test2 = load_image(Image.open(img2_dir))
    loss_func = FaceVerification(model, args.true_label, device=args.device)
    x1 = pytorch_switch(x_test1).detach()
    x2 = pytorch_switch(x_test2).detach()
    params = {
        "x1": x1.to(args.device),
        "x2": x2.to(args.device),
        "eps": args.s**2,
        "n_queries": args.queries,
        "save_directory": args.save_directory,
        "c": x1.shape[2],
        "h": x1.shape[0],
        "w": x1.shape[1],
        "N": args.N,
        "update_loc_period": args.li,
        "mut": args.mut,
        "temp": args.temp,
        "device": args.device
    }
    attack = Attack(params)
    attack.optimise(loss_func)

