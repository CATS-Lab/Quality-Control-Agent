import os
import json
import base64
import argparse

import numpy as np
import pandas as pd
from pathlib import Path

import smolagents

import open3d as o3d
from pypcd4 import PointCloud
import matplotlib.pyplot as plt

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# --------------------------
# Path configuration
# --------------------------
BASE_DIR = Path(__file__).parent
FRAMES_DIR = BASE_DIR / "Input" / "Frames"
POINTS_DIR = BASE_DIR / "Input" / "Points Clouds"
PROMPTS_DIR = BASE_DIR / "Input" / "Prompts"
RAG_DIR = BASE_DIR / "RAG"
OUTPUT_DIR = BASE_DIR / "Output"
API_KEYS_PATH = BASE_DIR / "configs"



# --------------------------
# Utility functions
# --------------------------
def encode_image(path: Path) -> str:
    """Convert image file to base64 string"""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
    

def point_cloud_to_depthmap(pcd_path: Path, save_path: Path, resolution=2048, use_intensity=False):
    """
    Convert raw point cloud (.pcd/.ply) into a bird-eye depth map image.
    
    Args:
        pcd_path (Path): Input point cloud file (.pcd or .ply).
        save_path (Path): Output image path (.png).
        resolution (int): Histogram bin resolution (default=2048).
        use_intensity (bool): Whether to weight by intensity instead of z.
    
    Returns:
        Path: Path to saved depthmap image.
    """
    # --------- Load with pypcd4 ---------
    pcd_raw = PointCloud.from_path(str(pcd_path))
    arr = pcd_raw.numpy()   # shape (N, F), columns: x,y,z,intensity,...
    
    if arr.shape[1] < 3:
        raise ValueError(f"{pcd_path} does not contain x,y,z columns.")

    x, y, z = arr[:, 0], arr[:, 1], arr[:, 2]
    if use_intensity and "intensity" in pcd_raw.fields:
        weights = arr[:, pcd_raw.fields.index("intensity")]
    else:
        weights = z

    # --------- Visualization with Open3D (optional) ---------
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(arr[:, :3])
    pcd.paint_uniform_color([0, 0, 1])

    # --------- Plot bird-eye depth map ---------
    plt.figure(figsize=(6, 6))
    plt.hist2d(x, y, bins=resolution, weights=weights, cmap="viridis")
    plt.colorbar(label="Height (z)" if not use_intensity else "Intensity")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Bird-eye Depth Map")
    plt.axis("equal")
    plt.savefig(save_path, dpi=200)
    plt.close()

    return save_path

def normalize_prompt(prompt_json):
    """
    Convert JSON-style prompt into smolagents-compatible message format.
    """
    role = prompt_json["role"]
    content_dict = prompt_json["content"]

    if isinstance(content_dict, list):
        return {"role": role, "content": content_dict}

    parts = []
    for key, value in content_dict.items():
        parts.append(f"{key.upper()}:\n{value}")
    text_content = "\n\n".join(parts)

    return {"role": role, "content": [{"type": "text", "text": text_content}]}
 
def load_prompts_report(query):
    """Load prompt files"""
    system_prompt = json.loads((PROMPTS_DIR / "Report" / "SYSTEM_Prompt.json").read_text(encoding="utf-8"))
    user_prompt = json.loads((PROMPTS_DIR / "Report" / "USER_Prompt.json").read_text(encoding="utf-8"))
    hidden_prompt = json.loads((PROMPTS_DIR / "Report" / "HIDDEN_Prompt.json").read_text(encoding="utf-8"))
    """Query prompt for RAG"""
    rag_prompt = load_references(query, top_k=3, max_results=5)
    
    return system_prompt, user_prompt, hidden_prompt, rag_prompt

def load_prompts_rag():
    """Load prompt files"""
    system_prompt = json.loads((PROMPTS_DIR / "RAG" / "SYSTEM_Prompt.json").read_text(encoding="utf-8"))
    user_prompt   = json.loads((PROMPTS_DIR / "RAG" / "USER_Prompt.json").read_text(encoding="utf-8"))
    hidden_prompt = json.loads((PROMPTS_DIR / "RAG" / "HIDDEN_Prompt.json").read_text(encoding="utf-8"))
    return system_prompt, user_prompt, hidden_prompt

def load_images():
    """Load frames and point cloud ply files"""
    # Load image frames
    frame_paths = list(FRAMES_DIR.glob("*.png"))
    frames = [encode_image(p) for p in frame_paths]

    # Convert point clouds to depth maps
    depthmaps = []
    for ply_file in POINTS_DIR.glob("*.ply"):
        depthmap_path = OUTPUT_DIR / f"{ply_file.stem}_depthmap.png"
        point_cloud_to_depthmap(ply_file, depthmap_path)
        depthmaps.append(encode_image(depthmap_path))

    return frames, depthmaps

def load_references(query_json: str, top_k: int = 3, max_results: int = 5) -> str:
    """
    Perform similarity search in the vector database for multiple queries and merge results.

    Args:
        query_json: A JSON string in the format {"queries": ["q1", "q2", ...]}.
        top_k: Number of results to return for each query.
        max_results: Maximum number of results to include in the final merged output 
                     (to avoid overly long context).

    Returns:
        A single string containing the merged retrieval results.
    """
    embedding_model = HuggingFaceEmbeddings(model_name="thenlper/gte-small")
    vectordb = FAISS.load_local(RAG_DIR / "vector_db", embeddings=embedding_model, allow_dangerous_deserialization=True)
    
    try:
        queries = json.loads(query_json).get("queries", [])
    except json.JSONDecodeError:
        raise ValueError("Input must be a JSON string, e.g., {\"queries\": [\"...\"]}")
    
    if not queries:
        raise ValueError("No 'queries' list found in JSON input")

    all_results = []
    seen_contents = set()
    
    for idx, query in enumerate(queries):
        # print(f"Executing query {idx+1}: {query}")
        results = vectordb.similarity_search(query, k=top_k)
        
        for result in results:
            if result.page_content not in seen_contents:
                seen_contents.add(result.page_content)
                all_results.append(result.page_content)
                
                # Stop if the maximum result limit is reached
                if len(all_results) >= max_results:
                    break
        if len(all_results) >= max_results:
            break
    
    # Format output
    combined_results = "\n\n".join([f"Reference {i+1}:\n{content}" for i, content in enumerate(all_results)])
    return combined_results

# --------------------------
# Initialize OpenAI client
# --------------------------
def initialize_client(config_path):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_name = config["model_name"]
    base_url = config["base_url"]
    api_key = config["api_key"]
    temperature = config["temperature"]

    # Initialize OpenAI client
    client = smolagents.OpenAIServerModel(model_id=model_name, api_base=base_url, api_key=api_key, temperature=temperature)
    return client

# --------------------------
# Call the language model
# --------------------------
def call_model_rag(system_prompt, user_prompt, hidden_prompt, frames, client):
    """
    Call the LLM with system, hidden, and user prompts plus optional images.
    
    Args:
        system_prompt (dict): Loaded SYSTEM_Prompt.json
        user_prompt (dict): Loaded USER_Prompt.json
        hidden_prompt (dict): Loaded HIDDEN_Prompt.json
        frames (list[str]): List of base64-encoded images
        client: LLM client instance
    
    Returns:
        str: Model output (should be JSON with query)
    """

    # Normalize system / hidden / user
    sys_msg = normalize_prompt(system_prompt)
    hid_msg = normalize_prompt(hidden_prompt)

    # Build user content (text + images)
    contents = [{"type": "text", "text": user_prompt["content"]["instruction"] + "\n\n" + user_prompt["content"]["description"] + "\n\n" + user_prompt["content"]["task"]}]
    
    for img_b64 in frames:
        contents.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
        })
        
    usr_msg = {"role": user_prompt["role"], "content": contents}

    # Messages stack
    messages = [sys_msg, hid_msg, usr_msg]

    # Call model
    response = client(messages=messages)

    return response.content

def call_model_report(system_prompt, user_prompt, hidden_prompt, rag_context, frames, depthmaps, client):
    """
    Call the LLM with system, hidden, and user prompts plus optional images.
    
    Args:
        system_prompt (dict): Loaded SYSTEM_Prompt.json
        user_prompt (dict): Loaded USER_Prompt.json
        hidden_prompt (dict): Loaded HIDDEN_Prompt.json
        frames (list[str]): List of base64-encoded images
        depthmaps (list[str]): List of base64-encoded depth maps
        client: LLM client instance
    
    Returns:
        str: Model output (should be JSON with query)
    """
    
    
    # Normalize system / hidden
    sys_msg = normalize_prompt(system_prompt)
    hid_msg = normalize_prompt(hidden_prompt)

    # Build user content (text + images + rag context)
    contents = [{"type": "text", "text": user_prompt["content"]["instruction"] + "\n\n" + user_prompt["content"]["description"] + "\n\n" + user_prompt["content"]["task"]}]
    
    if rag_context:
        contents.append({
            "type": "text",
            "text": f"Reference Materials (retrieved from Work Zone Manual):\n{rag_context}"
        })

    # Add image frames
    for idx, img_b64 in enumerate(frames, start=1):
        contents.append({"type": "text", "text": f"Frame {idx} (camera image):"})
        contents.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
        })

    # Add depth maps
    for idx, dm_b64 in enumerate(depthmaps, start=1):
        contents.append({"type": "text", "text": f"Depth Map {idx} (from point cloud):"})
        contents.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{dm_b64}"}
        })

    # Add user instruction and description at the end
    usr_msg = {"role": user_prompt["role"], "content": contents}
    
    # Messages stack
    messages = [sys_msg, hid_msg, usr_msg]
    
    # Call model
    response = client(messages=messages)

    return response.content

# --------------------------
# Save model outputs
# --------------------------
def save_query(text_output, model_name):
    model_output_dir = OUTPUT_DIR / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)
    (model_output_dir / "Query.txt").write_text(text_output, encoding="utf-8")


def save_output(text_output, model_name):
    # Make Directory for model if not exists
    model_output_dir = OUTPUT_DIR / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)
    # Save raw text output
    (model_output_dir / "Response.txt").write_text(text_output, encoding="utf-8")

# --------------------------
# Parse command-line arguments
# --------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(description="Parse model input arguments.")
    parser.add_argument("-m", "--model", type=str, required=True, help="Specify the model to use (e.g., deepseek, gpt-4, gpt-4-mini).")
    return parser.parse_args()

# --------------------------
# Main workflow
# --------------------------
if __name__ == "__main__":
    #### For RAG retrieval testing
    # args = parse_arguments()
    # model_name = args.model
    # config_path = API_KEYS_PATH / f"{model_name}.json"

    # system_prompt, user_prompt, hidden_prompt = load_prompts_rag()
    # frames, depthmaps = load_images()
    
    # client = initialize_client(config_path)
    # result = call_model_rag(system_prompt, user_prompt, hidden_prompt, frames, client)
    # save_output(result, model_name)
    
    
    
    #### For query generation testing
    # response_file = OUTPUT_DIR / "deepseek" / "Response.txt"
    # with open(response_file, "r", encoding="utf-8") as f:
    #     query_json = f.read()

    # print("Query Content：")
    # print(query_json)

    # combined_results = load_references(query_json, top_k=3, max_results=5)
    
    # print("\n=== Combined Results ===")
    # print(combined_results)
    
    
    
    ### For full report generation testing
    args = parse_arguments()
    model_name = args.model
    config_path = API_KEYS_PATH / f"{model_name}.json"

    response_file = OUTPUT_DIR / "deepseek" / "rag_query.txt"
    with open(response_file, "r", encoding="utf-8") as f:
        queries = f.read()
        
    system_prompt, user_prompt, hidden_prompt, rag_prompt = load_prompts_report(queries)
    frames, depthmaps = load_images()
    
    client = initialize_client(config_path)
    result = call_model_report(system_prompt, user_prompt, hidden_prompt, rag_prompt, frames, depthmaps, client)
    save_output(result, model_name)