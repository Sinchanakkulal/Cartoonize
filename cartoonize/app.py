import os
import io
import uuid
import sys
import yaml
import traceback
import subprocess

with open('./config.yaml', 'r') as fd:
    opts = yaml.safe_load(fd)

sys.path.insert(0, './white_box_cartoonizer/')

import cv2
from flask import Flask, render_template, make_response, flash # type: ignore
import flask # type: ignore
from PIL import Image
import numpy as np
import skvideo.io # type: ignore
if opts['colab-mode']:
    from flask_ngrok import run_with_ngrok # type: ignore

from cartoonize import WB_Cartoonize # type: ignore

if not opts['run_local']:
    if 'GOOGLE_APPLICATION_CREDENTIALS' in os.environ:
        from gcloud_utils import upload_blob, generate_signed_url, delete_blob, download_video
    else:
        raise Exception("GOOGLE_APPLICATION_CREDENTIALS not set in environment variables")
    from video_api import api_request
    import Algorithmia # type: ignore

app = Flask(__name__)
if opts['colab-mode']:
    run_with_ngrok(app)

app.config['UPLOAD_FOLDER_VIDEOS'] = 'static/uploaded_videos'
app.config['CARTOONIZED_FOLDER'] = 'static/cartoonized_images'
app.config['OPTS'] = opts

# Initialize Cartoonizer
wb_cartoonizer = WB_Cartoonize(os.path.abspath("white_box_cartoonizer/saved_models/"), opts['gpu'])

def convert_bytes_to_image(img_bytes):
    pil_image = Image.open(io.BytesIO(img_bytes))
    if pil_image.mode == "RGBA":
        image = Image.new("RGB", pil_image.size, (255, 255, 255))
        image.paste(pil_image, mask=pil_image.split()[3])
    else:
        image = pil_image.convert('RGB')
    return np.array(image)

def run_command(command):
    """Run a shell command with error handling."""
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr)
        raise Exception(f"Command failed: {command}")
    return result

@app.route('/')
@app.route('/cartoonize', methods=["POST", "GET"])
def cartoonize():
    opts = app.config['OPTS']
    if flask.request.method == 'POST':
        try:
            if flask.request.files.get('image'):
                img = flask.request.files["image"].read()
                image = convert_bytes_to_image(img)

                img_name = str(uuid.uuid4())
                cartoon_image = wb_cartoonizer.infer(image)
                
                cartoonized_img_name = os.path.join(app.config['CARTOONIZED_FOLDER'], img_name + ".jpg")
                cv2.imwrite(cartoonized_img_name, cv2.cvtColor(cartoon_image, cv2.COLOR_RGB2BGR))

                if not opts["run_local"]:
                    output_uri = upload_blob("cartoonized_images", cartoonized_img_name, img_name + ".jpg", content_type='image/jpg')
                    os.remove(cartoonized_img_name)
                    cartoonized_img_name = generate_signed_url(output_uri)

                return render_template("index_cartoonized.html", cartoonized_image=cartoonized_img_name)

            if flask.request.files.get('video'):
                filename = str(uuid.uuid4()) + ".mp4"
                video = flask.request.files["video"]
                original_video_path = os.path.join(app.config['UPLOAD_FOLDER_VIDEOS'], filename)
                video.save(original_video_path)

                modified_video_path = os.path.join(app.config['UPLOAD_FOLDER_VIDEOS'], filename.split(".")[0] + "_modified.mp4")

                file_metadata = skvideo.io.ffprobe(original_video_path)
                original_frame_rate = file_metadata.get('video', {}).get('@r_frame_rate', None)
                output_frame_rate = original_frame_rate if opts['original_frame_rate'] else opts['output_frame_rate']
                output_frame_rate_number = int(output_frame_rate.split('/')[0])

                width_resize = opts['resize-dim']

                trim_option = f"-t {opts['trim-video-length']}" if opts['trim-video'] else ""
                resolution_option = f"scale={width_resize}:-2" if not opts['original_resolution'] else "scale=-1:-2"

                run_command(f"ffmpeg -hide_banner -loglevel warning -ss 0 -i '{original_video_path}' {trim_option} -filter:v {resolution_option} -r {output_frame_rate_number} -c:a copy '{modified_video_path}'")

                audio_file_path = os.path.join(app.config['UPLOAD_FOLDER_VIDEOS'], filename.split(".")[0] + "_audio_modified.mp4")
                run_command(f"ffmpeg -hide_banner -loglevel warning -i '{modified_video_path}' -map 0:1 -vn -acodec copy -strict -2 '{audio_file_path}'")

                if opts["run_local"]:
                    cartoon_video_path = wb_cartoonizer.process_video(modified_video_path, output_frame_rate)
                else:
                    data_uri = upload_blob("processed_videos_cartoonize", modified_video_path, filename, content_type='video/mp4', algo_unique_key='cartoonizeinput')
                    response = api_request(data_uri)
                    delete_blob("processed_videos_cartoonize", filename)
                    cartoon_video_path = download_video('cartoonized_videos', os.path.basename(response['output_uri']), os.path.join(app.config['UPLOAD_FOLDER_VIDEOS'], filename.split(".")[0] + "_cartoon.mp4"))

                final_cartoon_video_path = os.path.join(app.config['UPLOAD_FOLDER_VIDEOS'], filename.split(".")[0] + "_cartoon_audio.mp4")
                run_command(f"ffmpeg -hide_banner -loglevel warning -i '{cartoon_video_path}' -i '{audio_file_path}' -codec copy -shortest '{final_cartoon_video_path}'")

                os.remove(original_video_path)
                os.remove(modified_video_path)
                os.remove(audio_file_path)
                os.remove(cartoon_video_path)

                return render_template("index_cartoonized.html", cartoonized_video=final_cartoon_video_path)

        except Exception as e:
            print(traceback.print_exc())
            flash("Our server hiccuped :/ Please upload another file! :)")
            return render_template("index_cartoonized.html")
    else:
        return render_template("index_cartoonized.html")

if __name__ == "__main__":
    if opts['colab-mode']:
        app.run()
    else:
        app.run(debug=False, host='127.0.0.1', port=int(os.environ.get('PORT', 8080)))
