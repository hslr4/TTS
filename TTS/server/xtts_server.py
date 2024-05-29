#!flask/bin/python
import argparse
import io
import os
from pathlib import Path
from threading import Lock
from TTS.tts.layers.xtts.xtts_manager import SpeakerManager
import torch
from TTS.utils.audio.numpy_transforms import save_wav
from TTS.tts.configs.xtts_config import XttsConfig

from flask import Flask, render_template, request, send_file

from TTS.utils.manage import ModelManager
from TTS.tts.models.xtts import Xtts


def create_argparser():
    def convert_boolean(x):
        return x.lower() in ["true", "1", "yes"]

    parser = argparse.ArgumentParser()

    # Args for running custom models
    parser.add_argument("--config_path", default=None, type=str, help="Path to model config file.")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to model file.",
    )
    parser.add_argument("--speaker_wav_path", type=str, default='/root/TTS/data/', help="path to look for speaker wavs for voice cloning")
    parser.add_argument("--speaker_wav", type=str, help="name of .wav file or directory of wav files in speaker_wav_path for voice cloning")
    parser.add_argument("--speaker_id", type=str, help="name of coqui voice")
    parser.add_argument("--speaker_embeddings_file", type=str, help="name of file path for custom embeddings")#, default='/root/TTS/data/custom_speakers_xtts.pth'
    parser.add_argument("--port", type=int, default=5002, help="port to listen on.")
    parser.add_argument("--use_cuda", type=convert_boolean, default=True, help="true to use CUDA.")
    parser.add_argument("--use_deepspeed", type=convert_boolean, default=False, help="true to use deepspeed.")
    parser.add_argument("--use_tensorrt", type=str, default='fp32', help="fp16, fp32 or True to use tensorrt, False otherwise.")
    parser.add_argument("--debug", type=convert_boolean, default=False, help="true to enable Flask debug mode.")
    parser.add_argument("--show_details", type=convert_boolean, default=False, help="Generate model detail page.")
    return parser

# parse the args
args = create_argparser().parse_args()

path = Path(__file__).parent / "../.models.json"
manager = ModelManager(path)

model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
model_path = args.model_path
config_path = None

if not model_path:
    model_path, config_path, model_item = manager.download_model(model_name)

use_tensorrt = False
if args.use_tensorrt == 'fp32':
    use_tensorrt = True
if args.use_tensorrt == 'fp16':
    use_tensorrt = 'fp16'

config = XttsConfig()
config.load_json(config_path)
model = Xtts.init_from_config(config)
model.load_checkpoint(
    config,
    checkpoint_path=os.path.dirname(model_path),
    speaker_file_path=args.speaker_embeddings_file,
    eval=True,
    use_deepspeed=args.use_deepspeed,
    use_tensorrt=use_tensorrt,
)
if args.use_cuda:
    model.cuda()

speaker_ids = list(model.speaker_manager.speakers.keys())
language_manager = getattr(model, "language_manager", None)

def load_voice(speaker_id=None, speaker_wav=None, **cloning_args):
    if not speaker_id and not speaker_wav:
        return None, None, 'no speaker_id or speaker_wav provided'
    if speaker_id and speaker_wav:
        print('speaker_id and speaker_wav are provided though only one can be used, using speaker_id')
    if speaker_id:
        if speaker_id not in speaker_ids:
            return None, None, 'invalid speaker_id provided'
        else:
            gpt_cond_latent, speaker_embedding = model.speaker_manager.speakers[speaker_id].values()
            return gpt_cond_latent, speaker_embedding, str(speaker_id)
    if speaker_wav:
        p = os.path.join(args.speaker_wav_path, speaker_wav)
        if not os.path.exists(p):
            return None, None, f'speaker_wav not found: {p}'
        else:
            if os.path.isdir(p):
                audio_path = []
                for f in os.listdir(p):
                    if os.path.isfile(os.path.join(p,f)): #and f.endswith('.wav')
                        audio_path.append(os.path.join(p,f))
                if len(audio_path) <= 0:
                    return None, None, f'no audio found in: {p}'
            else:
                audio_path=[p]
            gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=audio_path, **cloning_args)
            return gpt_cond_latent, speaker_embedding, str(speaker_wav)
        
gpt_cond_latent, speaker_embedding, v = load_voice(speaker_id=args.speaker_id, speaker_wav=args.speaker_wav)
if gpt_cond_latent is None or speaker_embedding is None:
    gpt_cond_latent, speaker_embedding, v = load_voice(speaker_id=speaker_ids[0])
print('using voice', v)


app = Flask(__name__)
lock = Lock()

@app.route("/")
def index():
    return render_template(
        "index.html",
        show_details=args.show_details,
        use_multi_speaker=True,
        use_multi_language=True,
        speaker_ids=speaker_ids,
        language_ids=language_manager.name_to_id if language_manager is not None else None,
        use_clone=True,
        pre_load_voice=True,
        show_gen_args=True,
        voice_file=args.speaker_embeddings_file or '/root/TTS/data/custom_speakers_xtts.pth'
    )


@app.route("/api/update_voice", methods=["GET", "POST"])
def update_voice():
    with lock:
        speaker_id = request.headers.get("speaker-id") or request.values.get("speaker_id", "")
        speaker_wav = request.headers.get("speaker-wav") or request.values.get("speaker_wav", "")
        int_args = ['gpt_cond_len', 'gpt_cond_chunk_len', 'max_ref_length', 'sound_norm_refs', 'librosa_trim_db']
        #gpt_cond_len=30, gpt_cond_chunk_len=4, max_ref_length=60 values from huggingface demo
        #max_ref_length=30,gpt_cond_len=6,gpt_cond_chunk_len=6,librosa_trim_db=None,sound_norm_refs=False #default values
        #num_gpt_outputs=1, gpt_cond_len=12, gpt_cond_chunk_len=4, max_ref_len=10, sound_norm_refs=False config default values
        cond_args = {}
        for a in int_args:
            v = request.headers.get(a.replace('_', '-')) or request.values.get(a, "")
            if v.isdigit():
                cond_args[a] = v
        sound_norm_refs = request.headers.get("sound-norm-refs") or request.values.get("sound_norm_refs", "")
        if sound_norm_refs:
            cond_args['sound_norm_refs'] = sound_norm_refs.lower() == 'true'
        try:
            latent, embedding, msg = load_voice(speaker_id, speaker_wav, **cond_args)
            if latent is not None and embedding is not None:
                global gpt_cond_latent, speaker_embedding
                gpt_cond_latent, speaker_embedding = latent, embedding
                return msg, 200
            else:
                return 'Failed to update voice:' + msg, 400
        except Exception as e:
            print(e)
            return 'Failed to update voice', 500

@app.route("/api/save_voice", methods=["GET", "POST"])
def save_voice():
    with lock:
        name = request.headers.get("name") or request.values.get("name", "")
        file = request.headers.get("file") or request.values.get("file") or args.speaker_embeddings_file
        if not path:
            return 'Failed to save voice, missing file', 400
        if not name:
            return 'Failed to save voice, missing name', 400
        if os.path.exists(file):
            speakers = torch.load(file)
        else:
            speakers = {}
        
        speakers[name] = {
            'gpt_cond_latent': gpt_cond_latent,
            'speaker_embedding': speaker_embedding
        }
        try:
            torch.save(speakers, file)
            model.speaker_manager = SpeakerManager(file)
            global speaker_ids
            speaker_ids = list(model.speaker_manager.speakers.keys())
            return f'saved current embeddings as {name} in {file}'
        except Exception as e:
            print(e)
            return 'Failed to save voice', 400


@app.route("/api/tts", methods=["GET", "POST"])
def tts():
    with lock:
        text = request.headers.get("text") or request.values.get("text", "")
        language_idx = request.headers.get("language-id") or request.values.get("language_id", "")

        print(f" > Model input: {text}")
        print(f" > Language Idx: {language_idx}")
        gen_arg_keys = ['temperature', 'repetition_penalty', 'length_penalty', 'top_k', 'top_p', 'do_sample']#'gpt_cond_len', 
        gen_args = {}
        for arg_name in gen_arg_keys:
            v = request.headers.get(arg_name.replace('_', '-')) or request.values.get(arg_name, "")
            if v:
                if arg_name == 'do_sample':
                    gen_args[arg_name] = v.lower() == 'true'
                elif arg_name == 'top_k':
                    gen_args[arg_name] = int(v)
                else:
                    gen_args[arg_name] = float(v)
            if 'do_sample' in gen_args and not gen_args['do_sample']:
                gen_args['num_beams'] = 1
        print(f" > Generation Args: {gen_args}")
        model_out = model.inference(
            text,
            language_idx,
            gpt_cond_latent,
            speaker_embedding,
            **gen_args
        )
        out = io.BytesIO()
        save_wav(wav=model_out['wav'], path=out, sample_rate=24000)
    return send_file(out, mimetype="audio/wav")

def main():
    app.run(debug=args.debug, host="::", port=args.port)


if __name__ == "__main__":
    main()