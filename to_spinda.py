from spindafy import SpindaConfig
from datetime import datetime
from PIL import Image, ImageOps
import random
import os
import shutil
import math
import json
from moviepy.editor import VideoFileClip, ImageSequenceClip
from multiprocessing import Pool, Manager
import numpy as np
import cv2
import time

###############
# Settings
###############
# Functions to call
STAGES = {
    "render_spindas": True, # If you don't want to modify the code you can disable this after your first run
    "filter_duplicate_spindas": True,  # If you don't want to modify the code you can disable this after your first run
    "split_mp4": True, # If you want to render the same video (e.g. to test different settings), you can disable this after the first split
    "render_frames": True, # The acutal rendering of each frame
    "output_video": True, # combine frames to a Video
    "add_audio": False # TODO: Implement! you can use     ffmpeg -i output_file.mp4 -i audio.mp3 -map 0 -map 1:a -c:v copy -shortest output_with_sound.mp4

}
SETTINGS = {
    "input_file": "res/test.mp4",
    "output_file": "res/test_output.mp4",
    "resolution": 16,
    "tri_color":  True,
    "keep_rendered_frames": False, # If you want to split rendering of a big video, keep this on. Just kill the process and restart when you want to continue
    "fps": 30 # Only relevant if STAGES["split_mp4"] == False
}

TEMP_DATA_PATH = "temp_files" 
# Default sub-image is 32x32. Lower to reduce calculation time. Values other than form 2**n aren't tested!
RESIZE_FACTOR = 0.25
# Only calculate every STEP_MULTIPLIER dot Value. At 1 there are 1021 Dots that each sub-image will be tested again (Duplicates will automatically be removed)
STEP_MULTIPLIER = 1
# Max amout of cached sub-images. Not rolling! Buffer size is checked every (2**18)//SHARE_BUFFER_SIZE frames
SHARE_BUFFER_SIZE = 2000
# Probably no need to be touched. Resolution up to which a shared dict will be used. Performance might depend on System. At higher resolutions the shaing costs more cpu than saved
SHARE_THRESHOLD = 16
# Threshold values for the Imagegeneration BI for the black/white split, TRI for the black(0 to TRI_THRESHOLD_LOW)/gray/white(TRI_THRESHOLD_HIGH to 255) split with shinys. 
BI_THRESHOLD = 128
TRI_THRESHOLD_LOW = 95
TRI_THRESHOLD_HIGH = 150
# NOT IMPLEMENTED; TODO: Mode for adaptive threshold, auto optimizing each frame for most details
THRESHOLD_AUTO = False

# Only touch if you change the Sprites
SPINDA_GAP = 25 # how close they will be pasted
SPINDA_SIZE = 32 # Full size (bounding box) of the sprite
SPOT_RANGES = [ # Haven't read the dot generation algorythm. Probably shouldn't change
    [0x0, 0x00000100, 0x1*STEP_MULTIPLIER],
    [0x00000100, 0x00010000, 0x00000100*STEP_MULTIPLIER],
    [0x00010000, 0x01000000, 0x00010000*STEP_MULTIPLIER],
    [0x01000000, 0x100000000, 0x01000000*STEP_MULTIPLIER]
]


def num_to_fixed_size_hex(num, size=8):
    hex_string = hex(num)[2:]
    hex_string = hex_string.zfill(size)
    return '0x' + hex_string

def render_spindas():
    mask_image = Image.open("res/spinda_visible_mask.png") # TODO: fix
    if not os.path.exists(TEMP_DATA_PATH):
        os.mkdir(TEMP_DATA_PATH)
    for i in range(4):
        if os.path.exists(os.path.join(TEMP_DATA_PATH, str(i))):
            shutil.rmtree(os.path.join(TEMP_DATA_PATH, str(i)))
        os.mkdir(os.path.join(TEMP_DATA_PATH, str(i)))
    for ranges in SPOT_RANGES:
        for i in range(ranges[0], ranges[1], ranges[2]):
            spinda = SpindaConfig.from_personality(i)
            img = spinda.render_dot(SPOT_RANGES.index(ranges))
            img = img.crop(img.getbbox())
            mask = Image.new("RGBA", img.size)
            mask.paste(mask_image, None)
            new = Image.new("RGBA", img.size)
            new.paste(img, None, mask)
            new = new.crop(new.getbbox())
            new.save(f"{TEMP_DATA_PATH}/{SPOT_RANGES.index(ranges)}/{num_to_fixed_size_hex(spinda.get_personality())}.png")

def filter_duplicate_spindas():
    def byte_compare(file1, file2):
        with open(file1, 'rb') as f1, open(file2, 'rb') as f2:
            return f1.read() == f2.read()
    for index in range(4):
        spindas = os.listdir(os.path.join(TEMP_DATA_PATH, str(index)))
        dupes = set()
        for i, base_spinda in enumerate(spindas):
            for dupe_spinda in spindas[i+1:]:
                if byte_compare(os.path.join(TEMP_DATA_PATH, str(index), base_spinda), os.path.join(TEMP_DATA_PATH, str(index), dupe_spinda)):
                    dupes.add(dupe_spinda)
        for dupe in dupes:
            os.remove(os.path.join(TEMP_DATA_PATH, str(index), dupe))
        print(f'{datetime.now()} Index {index} had {len(dupes)} dupes, removed them')

def calculate_score(image1:Image.Image, image2:Image.Image, tri_color):
    if not tri_color:
        score = 0
        pixels1 = image1.load()
        pixels2 = image2.load()
        if image1.size != image2.size:
            raise ValueError("Images dont have same size!")
        width, height = image1.size
        for y in range(height):
            for x in range(width):
                pixel2 = pixels2[x, y]
                if pixel2[3] != 0:
                    pixel1 = pixels1[x, y]
                    if pixel1[:3] == pixel2[:3]:
                        score += 1
                    else:
                        score -= 1
        return score
    else:        
        score = 0
        shiny_score = 0
        pixels1 = image1.load()
        pixels2 = image2.load()
        if image1.size != image2.size:
            raise ValueError("Images dont have same size!")
        width, height = image1.size

        for y in range(height):
            for x in range(width):
                pixel2 = pixels2[x, y]
                if pixel2[3] != 0:
                    pixel1 = pixels1[x, y]
                    if pixel1[:3] == pixel2[:3]:
                        score += 1
                    elif pixel1[:3] == (128, 128, 128):
                        shiny_score += 1
                    else:
                        score -= 1
                        shiny_score -= 1
        return score, shiny_score
        
def convert_to_spinda(i, spinda_images, image_input, output, resolution, calculated, tri_color, calculated_shiny={}):
    if not tri_color:
        last_time = datetime.now()
        img = Image.open(image_input)
        width, height = img.size
        img = img.resize([int(resolution*SPINDA_GAP), int(resolution*SPINDA_GAP*(height/width))])
        img = ImageOps.grayscale(img)
        img = img.point(lambda x: 0 if x < BI_THRESHOLD else 255, '1')
        img = img.convert("RGBA")
        width, height = img.size
        img_out = Image.new("RGB", img.size, (0, 0, 0))

        square_counter = 0
        calculated_counter = 0
        image_squares = []

        for y in range(0, height, SPINDA_GAP):
            for x in range(0, width, SPINDA_GAP):
                box = (x, y, x + SPINDA_SIZE, y + SPINDA_SIZE)
                image_squares.append(box)

        for box in image_squares:
            x, y, _, _ = box
            square = img.crop(box)
            new = Image.new("L", square.size, 0)
            new.paste(square, None, mask_image)
            square = new
            square = square.resize([int(square.size[0]*RESIZE_FACTOR), int(square.size[1]*RESIZE_FACTOR)])
            square = square.point(lambda x: 0 if x < BI_THRESHOLD else 255, '1')
            square = square.convert("RGBA")
            square_hash = square.convert("1").tobytes().hex()

            if square_hash in calculated.keys():
                best_spinda = calculated[square_hash]
            else:
                # Each spot gets separatly calculated for best effect, best spots get combined. TODO: Add check for overlapping Dots
                best_spinda = 0x00000000
                for ranges in SPOT_RANGES:
                    best_spot = 0x0
                    best_score = - math.inf
                    for i in range(ranges[0], ranges[1], ranges[2]):
                        if os.path.exists(os.path.join(TEMP_DATA_PATH, str(SPOT_RANGES.index(ranges)), f'{num_to_fixed_size_hex(i)}.png')):
                            score = calculate_score(square, spinda_images[num_to_fixed_size_hex(i)], tri_color)
                            if score > best_score:
                                best_score = score
                                best_spot = i
                    best_spinda = best_spinda ^ best_spot
                calculated[square_hash] = best_spinda
                calculated_counter += 1
            spinda = SpindaConfig.from_personality(best_spinda)
            bs_image = spinda.render_pattern()
            img_out.paste(bs_image, [x-SPINDA_SIZE//2, y-SPINDA_SIZE//2], mask=bs_image)
            square_counter += 1
            # print(f'{square_counter} of {spinda_pixels}', end='\r')
        img_out.save(output)
        print(f'{datetime.now()} Image {image_input} done, calculated {calculated_counter} of {square_counter} squares in {datetime.now() - last_time}')
    else:
        last_time = datetime.now()
        img = Image.open(image_input)
        width,height  = img.size
        img = img.resize([int(resolution*SPINDA_GAP), int(resolution*SPINDA_GAP*(height/width))])
        img = ImageOps.grayscale(img)
        img = img.point(lambda x: 0 if x < TRI_THRESHOLD_LOW else 128 if x < TRI_THRESHOLD_HIGH else 255, 'L')
        img = img.convert("RGBA")
        width, height = img.size
        img_out = Image.new("RGB", img.size, (0, 0, 0))

        square_counter = 0
        calculated_counter = 0
        image_squares = []

        for y in range(0, height, SPINDA_GAP):
            for x in range(0, width, SPINDA_GAP):
                box = (x, y, x + SPINDA_SIZE, y + SPINDA_SIZE)
                image_squares.append(box)

        for box in image_squares:
            x, y, _, _ = box
            square = img.crop(box)
            new = Image.new("L", square.size, 0)
            new.paste(square, None, mask_image)
            square = new
            square = square.resize([int(square.size[0]*RESIZE_FACTOR), int(square.size[1]*RESIZE_FACTOR)])
            square = square.point(lambda x: 0 if x < TRI_THRESHOLD_LOW else 128 if x < TRI_THRESHOLD_HIGH else 255, 'L')
            square = square.convert("RGBA")
            square_hash = square.convert("1").tobytes().hex()
            
            if square_hash in calculated.keys() and square_hash in calculated_shiny.keys():
                if (best_spinda_score:=calculated[square_hash][1]) < (best_spinda_score_shiny:=calculated_shiny[square_hash][1]):
                    best_spinda_shiny = calculated_shiny[square_hash][0]
                else:
                    best_spinda = calculated[square_hash][0]
            else:
                # Each spot gets separatly calculated for best effect, best spots get combined. TODO: Add check for overlapping Dots
                best_spinda = 0x00000000
                best_spinda_score = 0
                best_spinda_shiny = 0x00000000
                best_spinda_score_shiny = 0
                for ranges in SPOT_RANGES:
                    best_spot = 0x0
                    best_spot_shiny = 0x0
                    best_score = - math.inf
                    best_score_shiny = - math.inf
                    for i in range(ranges[0], ranges[1], ranges[2]):
                        if os.path.exists(os.path.join(TEMP_DATA_PATH, str(SPOT_RANGES.index(ranges)), f'{num_to_fixed_size_hex(i)}.png')):
                            score, shiny_score = calculate_score(square, spinda_images[num_to_fixed_size_hex(i)], tri_color)
                            if score > best_score:
                                best_score = score
                                best_spot = i
                            if shiny_score > best_score_shiny:
                                best_score_shiny = shiny_score
                                best_spot_shiny = i
                    best_spinda = best_spinda ^ best_spot
                    best_spinda_score += best_score
                    best_spinda_shiny = best_spinda_shiny ^ best_spot_shiny
                    best_spinda_score_shiny += best_score_shiny
                calculated[square_hash] = (best_spinda, best_spinda_score)
                calculated_shiny[square_hash] = (best_spinda_shiny, best_spinda_score_shiny)
                calculated_counter += 1
            if best_spinda_score < best_spinda_score_shiny:
                spinda = SpindaConfig.from_personality(best_spinda_shiny)
                bs_image = spinda.render_pattern(shiny=True)
            else:
                spinda = SpindaConfig.from_personality(best_spinda)
                bs_image = spinda.render_pattern()
            img_out.paste(bs_image, [x-SPINDA_SIZE//2, y-SPINDA_SIZE//2], mask=bs_image)
            square_counter += 1
            # print(f'{square_counter} of {spinda_pixels}', end='\r')
        img_out = img_out.resize((width, height))
        img_out.save(output)
        print(f'{datetime.now()} Image {image_input} done, calculated {calculated_counter} of {square_counter} squares in {datetime.now() - last_time}')

def split_mp4_to_images(video_path, output_folder):
    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    success, image = vidcap.read()
    count = 1
    while success:
        cv2.imwrite(f"{output_folder}/frame_{str(count).rjust(5, '0')}.jpg", image)    
        success, image = vidcap.read()
        print('Saved image ', count)
        count += 1
    return fps

def images_to_video(image_folder, output, fps):
    images = os.listdir(image_folder)
    img = Image.open(os.path.join(image_folder, images[0]))
    video=cv2.VideoWriter(output,-1,fps,img.size)
    for frame in images:
        video.write(cv2.imread(os.path.join(image_folder, frame)))

def handle_video(spinda_images, input_file, output_file, resolution, tri_color, keep_rendered_frames, fps):
    video_path = os.path.join(TEMP_DATA_PATH, "video")
    if not os.path.exists(video_path):
        os.mkdir(video_path)        
    edit_video_path = os.path.join(TEMP_DATA_PATH, "video_edit")
    if not os.path.exists(edit_video_path):
        os.mkdir(edit_video_path)
    fps = -1
    if STAGES["split_mp4"]:
        fps = split_mp4_to_images(input_file, video_path)
    if STAGES["render_frames"]:
        print(f'{datetime.now()} Starting frame rendering')
        frames = os.listdir(video_path)
        if keep_rendered_frames:            
            for rendered_frame in os.listdir(edit_video_path):
                frames.remove(rendered_frame)
        frame_share_size = (2**18)//SHARE_BUFFER_SIZE
        frame_batches = [frames[i:i+frame_share_size] for i in range(0, len(frames), frame_share_size)]
        print(f'{datetime.now()} {len(frames)} frames need to be rendered, {len(frame_batches)} batches of size {len(frame_batches[0])}')
        
        calculated = {}
        calculated_shiny = {}
        for j, batch in enumerate(frame_batches):
            pool = Pool()
            if resolution <= SHARE_THRESHOLD:
                print(f'{datetime.now()} Buffer has lenght {len(calculated.keys())}')
                if len(calculated.keys()) < SHARE_BUFFER_SIZE and len(calculated.keys()) > 0:
                    print(f'{datetime.now()} Buffer kept for next batch')
                else:
                    manager = Manager()
                    if tri_color:
                        calculated = manager.dict()
                        calculated_shiny = manager.dict()
                    else:
                        calculated = manager.dict()
                        calculated_shiny = {}

            for i, frame in enumerate(batch):
                pool.apply_async(convert_to_spinda, args=(i, spinda_images, os.path.join(video_path, frame), os.path.join(edit_video_path, frame), resolution, calculated, tri_color, calculated_shiny))
            
            pool.close()
            pool.join()
            print(f'{datetime.now()} Batch {j+1} of {len(frame_batches)} rendered')
    if STAGES["output_video"]:
        if fps == -1:
            fps = SETTINGS["fps"]
        images_to_video(edit_video_path, output_file, fps)

def get_spinda_images():
    spinda_images = {}
    for ranges in SPOT_RANGES:
        for i in range(ranges[0], ranges[1], ranges[2]):
            if os.path.exists(os.path.join(TEMP_DATA_PATH, str(SPOT_RANGES.index(ranges)), f'{num_to_fixed_size_hex(i)}.png')):
                spinda_image = Image.open(os.path.join(TEMP_DATA_PATH, str(SPOT_RANGES.index(ranges)), f'{num_to_fixed_size_hex(i)}.png'))
                spinda_image = spinda_image.resize([int(spinda_image.size[0]*RESIZE_FACTOR), int(spinda_image.size[1]*RESIZE_FACTOR)], Image.Resampling.NEAREST)
                spinda_images[num_to_fixed_size_hex(i)] = spinda_image
    return spinda_images

def main(input_file, output_file, resolution, tri_color, keep_rendered_frames, fps):
    start = datetime.now()
    if STAGES["render_spindas"]:
        print(f'{datetime.now()} Starting spinda rendering')
        render_spindas()
        print(f'{datetime.now()} Spindas were rendered')
    if STAGES["filter_duplicate_spindas"]:
        print(f'{datetime.now()} Starting removing duplicates')
        filter_duplicate_spindas()
        print(f'{datetime.now()} Duplicate Spindas were removed')

    spinda_images = get_spinda_images()

    if input_file.endswith(".png") or input_file.endswith(".jpg"):
        convert_to_spinda(0, spinda_images, input_file, output_file, resolution, {}, True)
    elif input_file.endswith(".mp4"):
        handle_video(spinda_images, input_file, output_file, resolution, tri_color, keep_rendered_frames, fps)
    else:
        print("Unhandled extension!")
    print(f'Finished at {datetime.now()} after {datetime.now()-start}')

mask_image = Image.open("res/spinda_overlap_mask.png")
mask_image = ImageOps.grayscale(mask_image)
threshold = 10
mask_image = mask_image.point(lambda x: 0 if x < threshold else 255, '1')

visible_mask = Image.open("res/spinda_visible_mask.png")
visible_mask = visible_mask.crop(visible_mask.getbbox())
visible_mask = ImageOps.grayscale(visible_mask)
threshold = 10
visible_mask = visible_mask.point(lambda x: 0 if x < threshold else 255, '1')

if __name__ == "__main__":
    main(**SETTINGS)
    