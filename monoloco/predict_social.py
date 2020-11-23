from datetime import datetime

import math
from contextlib import contextmanager

import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib.patches import Circle, FancyArrow
import torch
from PIL import Image

from pydub import AudioSegment
from pydub.playback import play

from .visuals.pifpaf_show import KeypointPainter, image_canvas
from .network import MonoLoco, PifPaf
from .network.process import factory_for_gt, preprocess_pifpaf, image_transform

from monoloco.multithread.VideoGet import VideoGet

def predict(args):
    cnt = 0
    # add args.device
    args.device = torch.device('cpu')
    if torch.cuda.is_available():
        args.device = torch.device('cuda')


    # load models
    args.camera = True
    camera_num = 0
    pifpaf = PifPaf(args)
    monoloco = MonoLoco(model=args.model, device=args.device)

    # Start recording
    print('Webcam On')
    video_getter = VideoGet(camera_num).start()

    while True:
        if video_getter.stopped:
            video_getter.stop()
            break

        frame = video_getter.frame

        image = cv2.resize(frame, None, fx=args.scale, fy=args.scale)
        height, width, _ = image.shape
        print('resized image size: {}'.format(image.shape))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        print('Image processing')
        processed_image_cpu = image_transform(image.copy())
        processed_image = processed_image_cpu.contiguous().to(args.device, non_blocking=True)
        fields = pifpaf.fields(torch.unsqueeze(processed_image, 0))[0]
        _, _, pifpaf_out = pifpaf.forward(image, processed_image_cpu, fields)

        key = cv2.waitKey(1)
        if key % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break

        pil_image = Image.fromarray(image)
        intrinsic_size = [xx * 1.3 for xx in pil_image.size]

        if args.output_directory is None:
            output_path = './data/output/'
        else:
            output_path = args.output_directory

        kk, dic_gt = factory_for_gt(intrinsic_size, path_gt=args.path_gt)
        image_t = image

        print('Run monoloco')
        # Run Monoloco
        boxes, keypoints = preprocess_pifpaf(pifpaf_out, intrinsic_size, enlarge_boxes=False)
        dic_out = monoloco.forward(keypoints, kk)
        dic_out = monoloco.post_process(dic_out, boxes, keypoints, kk, dic_gt, reorder=False)

        # Print
        file_name = str(datetime.now())
        show_social(args, image_t, output_path + file_name, pifpaf_out, dic_out)
        print('Image {}\n'.format(cnt) + '-' * 120)
        cnt += 1


    cv2.VideoCapture(camera_num).release()
    cv2.destroyAllWindows()

    # Option 2: Load json file of poses from PifPaf and run monoloco
    # else:
    #     for idx, im_path in enumerate(images):
    #
    #         # Load image
    #         with open(im_path, 'rb') as f:
    #             image = Image.open(f).convert('RGB')
    #         if args.output_directory is None:
    #             output_path = im_path
    #         else:
    #             file_name = os.path.basename(im_path)
    #             output_path = os.path.join(args.output_directory, file_name)
    #
    #         im_size = (float(image.size[0] / args.scale),
    #                    float(image.size[1] / args.scale))  # Width, Height (original)
    #         kk, dic_gt = factory_for_gt(im_size, name=im_path, path_gt=args.path_gt)
    #         image_t = torchvision.transforms.functional.to_tensor(image).permute(1, 2, 0)
    #
    #         # Load json
    #         basename, ext = os.path.splitext(os.path.basename(im_path))
    #         extension = ext + '.pifpaf.json'
    #         path_json = os.path.join(args.json_dir, basename + extension)
    #         annotations = open_annotations(path_json)
    #
    #         # Run Monoloco
    #         boxes, keypoints = preprocess_pifpaf(annotations, im_size, enlarge_boxes=False)
    #         dic_out = monoloco.forward(keypoints, kk)
    #         dic_out = monoloco.post_process(dic_out, boxes, keypoints, kk, dic_gt, reorder=False)
    #
    #         # Print
    #         show_social(args, image_t, output_path, annotations, dic_out)
    #
    #         print('Image {}\n'.format(cnt) + '-' * 120)
    #         cnt += 1


def show_social(args, image_t, output_path, annotations, dic_out):
    """Output frontal image with poses or combined with bird eye view"""

    assert 'front' in args.output_types or 'bird' in args.output_types, "outputs allowed: front and/or bird"
    #warning sounds
    song = AudioSegment.from_mp3("data/sounds/warning.mp3")

    angles = dic_out['angles']
    xz_centers = [[xx[0], xx[2]] for xx in dic_out['xyz_pred']]
    image_t = putdatetime(image_t, datetime.now())

    colors = ['r' if social_distance(xz_centers, angles, idx) else 'deepskyblue'
              for idx, _ in enumerate(dic_out['xyz_pred'])]

    if colors == 'r':
        cv2.imwrite(output_path + '_violate.png', image_t)
        print('Violation is detected')
        play(song)

    if 'front' and 'bird' in args.output_types:
        # Prepare colors
        keypoint_sets, scores = get_pifpaf_outputs(annotations)
        uv_centers = dic_out['uv_heads']
        sizes = [abs(dic_out['uv_heads'][idx][1] - uv_s[1]) / 1.5 for idx, uv_s in enumerate(dic_out['uv_shoulders'])]

        keypoint_painter = KeypointPainter(show_box=False)
        with image_canvas(image_t,
                          output_path + '_front.png',
                          show=args.show,
                          fig_width=10,
                          dpi_factor=1.0) as ax:
            keypoint_painter.keypoints(ax, keypoint_sets, colors=colors)
            draw_orientation(ax, uv_centers, sizes, angles, colors, mode='front')

            with bird_canvas(args, output_path, show=args.show) as ax1:
                draw_orientation(ax1, xz_centers, [], angles, colors, mode='bird')
                play(song)

    else:
        if 'front' in args.output_types:
            # Prepare colors
            keypoint_sets, scores = get_pifpaf_outputs(annotations)
            uv_centers = dic_out['uv_heads']
            sizes = [abs(dic_out['uv_heads'][idx][1] - uv_s[1]) / 1.5 for idx, uv_s in enumerate(dic_out['uv_shoulders'])]

            keypoint_painter = KeypointPainter(show_box=False)
            with image_canvas(image_t,
                              output_path + '_front.png',
                              show=args.show,
                              fig_width=10,
                              dpi_factor=1.0) as ax:
                keypoint_painter.keypoints(ax, keypoint_sets, colors=colors)
                draw_orientation(ax, uv_centers, sizes, angles, colors, mode='front')


        if 'bird' in args.output_types:
            with bird_canvas(args, output_path, show=args.show) as ax1:
                draw_orientation(ax1, xz_centers, [], angles, colors, mode='bird')


def draw_orientation(ax, centers, sizes, angles, colors, mode):

    if mode == 'front':
        length = 5
        fill = False
        alpha = 0.6
        zorder_circle = 0.5
        zorder_arrow = 5
        linewidth = 1.5
        edgecolor = 'k'
        radiuses = [s / 1.2 for s in sizes]
    else:
        length = 1.3
        head_width = 0.3
        linewidth = 2
        radiuses = [0.2] * len(centers)
        fill = True
        alpha = 1
        zorder_circle = 2
        zorder_arrow = 1

    for idx, theta in enumerate(angles):
        color = colors[idx]
        radius = radiuses[idx]

        if mode == 'front':
            x_arr = centers[idx][0] + (length + radius) * math.cos(theta)
            z_arr = length + centers[idx][1] + (length + radius) * math.sin(theta)
            delta_x = math.cos(theta)
            delta_z = math.sin(theta)
            head_width = max(10, radiuses[idx] / 1.5)

        else:
            edgecolor = color
            x_arr = centers[idx][0]
            z_arr = centers[idx][1]
            delta_x = length * math.cos(theta)
            delta_z = - length * math.sin(theta)  # keep into account kitti convention

        circle = Circle(centers[idx], radius=radius, color=color, fill=fill, alpha=alpha, zorder=zorder_circle)
        arrow = FancyArrow(x_arr, z_arr, delta_x, delta_z, head_width=head_width, edgecolor=edgecolor,
                           facecolor=color, linewidth=linewidth, zorder=zorder_arrow)
        ax.add_patch(circle)
        ax.add_patch(arrow)


def social_distance(centers, angles, idx, threshold=2.5):
    """
    return flag of alert if social distancing is violated
    """
    xx = centers[idx][0]
    zz = centers[idx][1]
    angle = angles[idx]
    distances = [math.sqrt((xx - centers[i][0]) ** 2 + (zz - centers[i][1]) ** 2) for i, _ in enumerate(centers)]
    sorted_idxs = np.argsort(distances)

    for i in sorted_idxs[1:]:

        # First check for distance
        if distances[i] > threshold:
            return False

        # More accurate check based on orientation and future position
        elif check_social_distance((xx, centers[i][0]), (zz, centers[i][1]), (angle, angles[i])):
            return True

    return False


def check_social_distance(xxs, zzs, angles):
    """
    Violation if same angle or ine in front of the other
    Obtained by assuming straight line, constant velocity and discretizing trajectories
    """
    min_distance = 0.5
    theta0 = angles[0]
    theta1 = angles[1]
    steps = np.linspace(0, 2, 20)  # Discretization 20 steps in 2 meters
    xs0 = [xxs[0] + step * math.cos(theta0) for step in steps]
    zs0 = [zzs[0] - step * math.sin(theta0) for step in steps]
    xs1 = [xxs[1] + step * math.cos(theta1) for step in steps]
    zs1 = [zzs[1] - step * math.sin(theta1) for step in steps]
    distances = [math.sqrt((xs0[idx] - xs1[idx]) ** 2 + (zs0[idx] - zs1[idx]) ** 2) for idx, _ in enumerate(xs0)]
    if np.min(distances) <= max(distances[0] / 1.5, min_distance):
        return True
    return False


def calculate_margin(distance):
    """TOOD: Be permissive in orientation for far people and less for very close ones"""
    margin = distance * 2 / 5
    return margin


def get_pifpaf_outputs(annotations):
    """Extract keypoints sets and scores from output dictionary"""
    if not annotations:
        return [], []
    keypoints_sets = np.array([dic['keypoints'] for dic in annotations]).reshape(-1, 17, 3)
    score_weights = np.ones((keypoints_sets.shape[0], 17))
    score_weights[:, 3] = 3.0
    # score_weights[:, 5:] = 0.1
    # score_weights[:, -2:] = 0.0  # ears are not annotated
    score_weights /= np.sum(score_weights[0, :])
    kps_scores = keypoints_sets[:, :, 2]
    ordered_kps_scores = np.sort(kps_scores, axis=1)[:, ::-1]
    scores = np.sum(score_weights * ordered_kps_scores, axis=1)
    return keypoints_sets, scores

@contextmanager
def bird_canvas(args, output_path, show=True):
    fig, ax = plt.subplots(1, 1)
    fig.set_tight_layout(True)
    output_path = output_path + '_bird.png'
    x_max = args.z_max / 1.5
    ax.plot([0, x_max], [0, args.z_max], 'k--')
    ax.plot([0, -x_max], [0, args.z_max], 'k--')
    ax.set_ylim(0, args.z_max + 1)
    yield ax

    fig.savefig(output_path)
    if show:
        fig.canvas.flush_events()
        fig.canvas.draw()
        plt.pause(0.001)
    plt.close(fig)
    print('Bird-eye-view image saved')

def putdatetime(frame, time):
    """
    Add iterations per second text to lower-left corner of a frame.
    """
    cv2.putText(frame, "{:.0f}".format(time),
        (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
    return frame