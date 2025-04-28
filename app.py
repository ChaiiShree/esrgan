import fix_torch
import streamlit as st
import os
import subprocess
from PIL import Image
import torch
import time
import numpy as np
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

# Set page configuration
st.set_page_config(
    page_title="ESRGAN",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="auto",
)

# Cache function to load images
@st.cache_data
def load_image(image_file):
    img = Image.open(image_file)
    return img

# Cache function to load model
@st.cache_resource
def load_model(model_name):
    """Load the selected ESRGAN model"""
    if model_name == "ESRGAN_cheti":
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        model_path = 'weights/RealESRGAN_x4plus.pth'
    elif model_name == "RealESRGAN_x4plus_anime_6B":
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        model_path = 'weights/RealESRGAN_x4plus_anime_6B.pth'
    elif model_name == "realesr-general-x4v3":
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        model_path = 'weights/realesr-general-x4v3.pth'
    
    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=400,  # Use tile for large images to avoid memory issues
        tile_pad=10,
        pre_pad=0,
        half=torch.cuda.is_available()  # Use half precision if CUDA is available
    )
    
    return upsampler

def enhance_image(input_path, output_path, model_name, face_enhance=False, progress_callback=None):
    """Enhance image using ESRGAN with progress updates"""
    # Update progress - Model loading
    if progress_callback:
        progress_callback(0.1, "Loading model...")
    
    upsampler = load_model(model_name)
    
    # Update progress - Image loading
    if progress_callback:
        progress_callback(0.2, "Loading image...")
    
    # Read image
    img = Image.open(input_path).convert('RGB')
    img_array = np.array(img)
    
    # Update progress - Processing started
    if progress_callback:
        progress_callback(0.3, "Starting enhancement...")
    
    # Process image
    if face_enhance:
        # Update progress - Face enhancement preparation
        if progress_callback:
            progress_callback(0.4, "Setting up face enhancement...")
        
        # Import GFPGAN for face enhancement
        from gfpgan import GFPGANer
        face_enhancer = GFPGANer(
            model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
            upscale=4,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=upsampler
        )
        
        # Update progress - Face enhancement processing
        if progress_callback:
            progress_callback(0.5, "Enhancing faces...")
        
        _, _, output = face_enhancer.enhance(img_array, has_aligned=False, only_center_face=False, paste_back=True)
    else:
        # Update progress - Super-resolution processing
        if progress_callback:
            progress_callback(0.5, "Applying super-resolution...")
        
        output, _ = upsampler.enhance(img_array, outscale=4)
    
    # Update progress - Post-processing
    if progress_callback:
        progress_callback(0.8, "Finalizing image...")
    
    # Save result
    output_img = Image.fromarray(output)
    output_img.save(output_path)
    
    # Update progress - Complete
    if progress_callback:
        progress_callback(1.0, "Enhancement complete!")
    
    return output_img

def main():
    # Header
    st.title("✨ Image Super Resolution using ESRGAN")
    st.markdown("Upload your low-resolution images and enhance them with AI!")
    
    # Model selection
    model_name = st.selectbox(
        "Select Model:",
        ("ESRGAN_cheti")
    )
    
    # Model descriptions
    model_descriptions = {
        "ESRGAN_cheti": "General purpose model for photos (4x upscaling)"
    }
    
    st.info(model_descriptions[model_name])
    
    # Face enhancement option
    face_enhance = st.checkbox("Enable Face Enhancement", value=False)
    if face_enhance:
        st.write("Face enhancement will improve facial details in portraits")
    
    # Performance warning
    st.warning("⚠️ Enhancement may take several minutes depending on image size and your computer's specifications. Larger images will take longer to process.")
    
    # File uploader
    st.markdown("### Upload Image")
    st.info('✨ Supports PNG, JPG, JPEG image formats')
    uploaded_file = st.file_uploader("Choose an image file...", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        # Display original image
        original_image = load_image(uploaded_file)
        st.image(original_image, caption="Original Image", use_column_width=True)
        
        # Image info
        width, height = original_image.size
        st.text(f"Image dimensions: {width} × {height} pixels")
        
        # Save uploaded file
        os.makedirs("uploads", exist_ok=True)
        os.makedirs("outputs", exist_ok=True)
        input_path = os.path.join("uploads", uploaded_file.name)
        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Output path
        file_name, file_ext = os.path.splitext(uploaded_file.name)
        output_path = os.path.join("outputs", f"{file_name}_enhanced{file_ext}")
        
        # Process button
        if st.button("Enhance Image"):
            # Create progress bar and status text
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(progress, status):
                progress_bar.progress(progress)
                status_text.text(status)
            
            try:
                # Initial status
                update_progress(0, "Preparing...")
                
                # Process image with progress updates
                enhanced_image = enhance_image(
                    input_path, 
                    output_path, 
                    model_name, 
                    face_enhance, 
                    progress_callback=update_progress
                )
                
                # Display result
                st.success("Enhancement complete!")
                st.image(enhanced_image, caption="Enhanced Image", use_column_width=True)
                
                # Image comparison
                st.subheader("Before & After Comparison")
                col1, col2 = st.columns(2)
                with col1:
                    st.image(original_image, caption="Original")
                with col2:
                    st.image(enhanced_image, caption="Enhanced")
                
                # Download button
                with open(output_path, "rb") as file:
                    st.download_button(
                        label="Download Enhanced Image",
                        data=file,
                        file_name=f"enhanced_{uploaded_file.name}",
                        mime=f"image/{file_ext[1:]}"
                    )
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
                st.info("Try with a smaller image or check if the model files are downloaded correctly.")
    else:
        st.warning("Please upload an image to enhance")
    
    # Footer
    st.markdown("---")
    st.markdown("Powered by ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks")
    st.markdown("Processing times depend on your hardware. CPU processing is significantly slower than GPU.")

if __name__ == "__main__":
    import numpy as np
    main()
