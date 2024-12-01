import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pytesseract
from PIL import Image
import PyPDF2
import io

# Specify the model
MODEL_NAME = "google/gemma-2b-it"

class LegalEaseAssistant:
    def __init__(self, model_name=MODEL_NAME):
        # Load tokenizer and model with specific optimizations for Spaces
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map="cpu",  # Explicitly use CPU for Spaces compatibility
            load_in_8bit=True,  # Use 8-bit quantization to reduce memory usage
            torch_dtype=torch.float16
        )
    
    def extract_text_from_input(self, input_file):
        """
        Extract text from different input types
        Supports plain text, images (via OCR), and PDFs
        """
        # If input is a string, assume it's plain text
        if isinstance(input_file, str):
            return input_file
        
        # If input is an image file
        if isinstance(input_file, Image.Image):
            try:
                # Use pytesseract for OCR
                return pytesseract.image_to_string(input_file)
            except Exception as e:
                return f"Error extracting text from image: {str(e)}"
        
        # If input is a PDF file
        if hasattr(input_file, 'name') and input_file.name.lower().endswith('.pdf'):
            try:
                # Use PyPDF2 to extract text
                pdf_reader = PyPDF2.PdfReader(input_file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n\n"
                return text
            except Exception as e:
                return f"Error extracting text from PDF: {str(e)}"
        
        # If we can't process the input
        return "Unsupported input type"
    
    def generate_response(self, input_file, task_type):
        """Generic method to generate responses for different tasks"""
        # Extract text first
        text = self.extract_text_from_input(input_file)
        
        task_prompts = {
            "simplify": f"Simplify the following legal text in clear, plain language:\n\n{text}\n\nSimplified explanation:",
            "summary": f"Provide a concise summary of the following legal document:\n\n{text}\n\nSummary:",
            "key_terms": f"Identify and explain the key legal terms and obligations in this text:\n\n{text}\n\nKey Terms:",
            "risk": f"Perform a risk analysis on the following legal document:\n\n{text}\n\nRisk Assessment:"
        }
        
        prompt = task_prompts.get(task_type, f"Analyze the following text:\n\n{text}\n\nAnalysis:")
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # Generate response with controlled generation
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=300,  # Limit token generation
            num_return_sequences=1,
            do_sample=True,  # Add some randomness
            temperature=0.7,  # Control creativity
            top_p=0.9  # Nucleus sampling
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the relevant part of the response
        response_parts = response.split(prompt.split("\n\n")[-1])
        return response_parts[-1].strip() if len(response_parts) > 1 else response.strip()
    
    def compare_contracts(self, contract1, contract2):
        """Compare two contracts"""
        # Extract text from inputs
        text1 = self.extract_text_from_input(contract1)
        text2 = self.extract_text_from_input(contract2)
        
        prompt = f"""Compare these two legal documents, highlighting key differences and similarities:

Contract 1:
{text1}

Contract 2:
{text2}

Comparison Analysis:"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=400,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        
        comparison = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return comparison.split("Comparison Analysis:")[-1].strip()

# Create Gradio Interface
def create_interface():
    with gr.Blocks(title="LegalEase: AI Legal Assistant") as demo:
        gr.Markdown("# 📜 LegalEase: AI-Powered Legal Document Assistant")
        
        with gr.Tabs():
            # Simplify Language Tab
            with gr.Tab("Simplify Language"):
                with gr.Row():
                    simplify_input = gr.File(
                        file_types=['txt', 'pdf', 'image'], 
                        label="Upload Legal Document (Text/PDF/Image)"
                    )
                    simplify_text_input = gr.Textbox(label="Or Paste Text", lines=3)
                simplify_output = gr.Textbox(label="Simplified Explanation", lines=6)
                simplify_btn = gr.Button("Simplify Language")
                
                def simplify_handler(file, text):
                    # Prioritize file input if provided
                    input_source = file or text
                    return LegalEaseAssistant().generate_response(input_source, "simplify") if input_source else ""
                
                simplify_btn.click(
                    fn=simplify_handler,
                    inputs=[simplify_input, simplify_text_input],
                    outputs=simplify_output
                )
            
            # Document Summary Tab
            with gr.Tab("Document Summary"):
                with gr.Row():
                    summary_input = gr.File(
                        file_types=['txt', 'pdf', 'image'], 
                        label="Upload Legal Document (Text/PDF/Image)"
                    )
                    summary_text_input = gr.Textbox(label="Or Paste Text", lines=3)
                summary_output = gr.Textbox(label="Document Summary", lines=6)
                summary_btn = gr.Button("Generate Summary")
                
                def summary_handler(file, text):
                    input_source = file or text
                    return LegalEaseAssistant().generate_response(input_source, "summary") if input_source else ""
                
                summary_btn.click(
                    fn=summary_handler,
                    inputs=[summary_input, summary_text_input],
                    outputs=summary_output
                )
            
            # Key Terms Tab
            with gr.Tab("Key Terms"):
                with gr.Row():
                    terms_input = gr.File(
                        file_types=['txt', 'pdf', 'image'], 
                        label="Upload Legal Document (Text/PDF/Image)"
                    )
                    terms_text_input = gr.Textbox(label="Or Paste Text", lines=3)
                terms_output = gr.Textbox(label="Key Terms and Obligations", lines=6)
                terms_btn = gr.Button("Highlight Key Terms")
                
                def terms_handler(file, text):
                    input_source = file or text
                    return LegalEaseAssistant().generate_response(input_source, "key_terms") if input_source else ""
                
                terms_btn.click(
                    fn=terms_handler,
                    inputs=[terms_input, terms_text_input],
                    outputs=terms_output
                )
            
            # Contract Comparison Tab
            with gr.Tab("Contract Comparison"):
                with gr.Row():
                    compare_input1 = gr.File(
                        file_types=['txt', 'pdf', 'image'], 
                        label="Contract 1 (Text/PDF/Image)"
                    )
                    compare_text1 = gr.Textbox(label="Or Paste Contract 1 Text", lines=3)
                with gr.Row():
                    compare_input2 = gr.File(
                        file_types=['txt', 'pdf', 'image'], 
                        label="Contract 2 (Text/PDF/Image)"
                    )
                    compare_text2 = gr.Textbox(label="Or Paste Contract 2 Text", lines=3)
                compare_output = gr.Textbox(label="Comparison Analysis", lines=6)
                compare_btn = gr.Button("Compare Contracts")
                
                def compare_handler(file1, text1, file2, text2):
                    input_source1 = file1 or text1
                    input_source2 = file2 or text2
                    return LegalEaseAssistant().compare_contracts(input_source1, input_source2) if (input_source1 and input_source2) else ""
                
                compare_btn.click(
                    fn=compare_handler,
                    inputs=[compare_input1, compare_text1, compare_input2, compare_text2],
                    outputs=compare_output
                )
            
            # Risk Analysis Tab
            with gr.Tab("Risk Analysis"):
                with gr.Row():
                    risk_input = gr.File(
                        file_types=['txt', 'pdf', 'image'], 
                        label="Upload Legal Document (Text/PDF/Image)"
                    )
                    risk_text_input = gr.Textbox(label="Or Paste Text", lines=3)
                risk_output = gr.Textbox(label="Risk Assessment", lines=6)
                risk_btn = gr.Button("Analyze Risks")
                
                def risk_handler(file, text):
                    input_source = file or text
                    return LegalEaseAssistant().generate_response(input_source, "risk") if input_source else ""
                
                risk_btn.click(
                    fn=risk_handler,
                    inputs=[risk_input, risk_text_input],
                    outputs=risk_output
                )
    
    return demo

# Create the interface
demo = create_interface()

# Launch the app
if __name__ == "__main__":
    demo.launch()