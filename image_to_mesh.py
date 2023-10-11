# image_to_3dmesh.py

import numpy as np
import open3d as o3d
import requests
from matplotlib import pyplot as plt
from PIL import Image
from io import BytesIO
import torch
from transformers import GLPNImageProcessor, GLPNForDepthEstimation


def get_image_from_url(url):
    response = requests.get(url)
    pil_image = Image.open(BytesIO(response.content))
    return pil_image


def get_resized_dimensions(image, max_height=480, divisor=32):
    new_height = min(image.height, max_height)
    new_height -= (new_height % divisor)
    new_width = int(new_height * image.width / image.height)
    remainder = new_width % divisor
    if remainder < divisor / 2:
        new_width -= remainder
    else:
        new_width += divisor - remainder
    return new_width, new_height


def prepare_image(image, feature_extractor):
    new_dims = get_resized_dimensions(image)
    resized_image = image.resize(new_dims)
    return feature_extractor(images=resized_image, return_tensors="pt"), new_dims


def get_prediction(inputs, model):
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.predicted_depth


def remove_borders(image, predicted_depth, pad=16):
    output = predicted_depth.squeeze().cpu().numpy() * 1000.0
    output_cropped = output[pad:-pad, pad:-pad]
    image_cropped = image.crop((pad, pad, image.width - pad, image.height - pad))
    return image_cropped, output_cropped


def visualize(image, output):
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(image)
    ax[0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax[1].imshow(output, cmap='plasma')
    ax[1].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.tight_layout()
    plt.show()


def get_rgbd_image(image, depth_output):
    normalized_depth = (depth_output * 255 / np.max(depth_output)).astype('uint8')
    color_o3d = o3d.geometry.Image(np.array(image))
    depth_o3d = o3d.geometry.Image(normalized_depth)
    return o3d.geometry.RGBDImage.create_from_color_and_depth(color_o3d, depth_o3d, convert_rgb_to_intensity=False)


def get_camera_intrinsics(width, height, fx=500, fy=500):
    intrinsics = o3d.camera.PinholeCameraIntrinsic()
    intrinsics.set_intrinsics(width, height, fx, fy, width/2, height/2)
    return intrinsics


def create_point_cloud_from_rgbd(color_image, depth_output, camera_intrinsics):
    rgbd_image = get_rgbd_image(color_image, depth_output)
    return o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsics)


def generate_3d_mesh(point_cloud):
    cl, ind = point_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=20.0)
    cleaned_point_cloud = point_cloud.select_by_index(ind)
    cleaned_point_cloud.estimate_normals()
    cleaned_point_cloud.orient_normals_to_align_with_direction()
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(cleaned_point_cloud, depth=10, n_threads=1)[0]
    rotation = mesh.get_rotation_matrix_from_xyz((np.pi, 0, 0))
    mesh.rotate(rotation, center=(0, 0, 0))
    return mesh


def main():
    # Initialize pre-trained depth estimation model and feature extractor
    feature_extractor = GLPNImageProcessor.from_pretrained("vinvino02/glpn-nyu")
    model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-nyu")
    url = "https://www.ikea.com/it/it/images/products/slattum-struttura-letto-imbottita-knisa-grigio-chiaro__0768244_pe754388_s5.jpg?f=xl"

    # Depth estimation
    image = get_image_from_url(url)
    inputs, new_dims = prepare_image(image, feature_extractor)
    predicted_depth = get_prediction(inputs, model)
    image_cropped, output_cropped = remove_borders(image.resize(new_dims), predicted_depth, pad=16)
    visualize(image_cropped, output_cropped)

    # Point cloud construction
    width, height = image_cropped.size
    intrinsics = get_camera_intrinsics(width, height)
    point_cloud = create_point_cloud_from_rgbd(image_cropped, output_cropped, intrinsics)

    # Generate 3D mesh
    mesh = generate_3d_mesh(point_cloud)
    o3d.io.write_triangle_mesh('./mesh.obj', mesh)
    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

if __name__ == "__main__":
    main()
