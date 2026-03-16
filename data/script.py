# import json
# import os

# # Paths
# INPUT_FILE = "data/raw/quotes_master.jsonl"
# OUTPUT_FILE = "data/raw/quotes_master_fixed.jsonl"
# IMAGE_SUBFOLDER = "images" 

# fixed_count = 0

# with open(INPUT_FILE, 'r', encoding='utf-8') as f_in, \
#      open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
    
#     for line in f_in:
#         row = json.loads(line)
        
#         if row.get("modality") == "image":
#             # Extract the hash from the quote_id
#             # Example quote_id: "train_L1_image_1" or a hash-based ID
#             # If your quote_id looks like "0b85477..._image6", we split it:
#             quote_id = row.get("quote_id", "")
#             tag = row.get("image_tag", "") # e.g., "image6"
            
#             # Logic: We need to find the document hash. 
#             # In your data, the line_idx or a specific field usually maps to this.
#             # Based on your example, the format is: {doc_hash}_{tag}.jpg
            
#             # Since I don't have the hash mapping, we can try to find the file 
#             # that ends with _{tag}.jpg in your directory if the hash isn't in the JSON.
#             # BUT, usually the 'quote_id' contains the info we need.
            
#             # Let's assume the hash is derived from the 'quote_id' logic.
#             # REVISED LOGIC: If the image path is missing, we construct it.
#             # If your files are named like '0b85477387a9d0cc33fca0f4becaa0e5_image6.jpg'
            
#             # Check if quote_id has the hash:
#             doc_hash = quote_id.split('_')[0] 
#             reconstructed_path = f"{IMAGE_SUBFOLDER}/{doc_hash}_{tag}.jpg"
            
#             row["image_path"] = reconstructed_path
#             fixed_count += 1
            
#         f_out.write(json.dumps(row) + "\n")

# print(f"✅ Repair Complete! Fixed {fixed_count} entries.")
# print(f"Example path created: {reconstructed_path}")
import json
import os

# Paths
INPUT_FILE = "data/processed/quotes_master.jsonl"
OUTPUT_FILE = "data/processed/quotes_master_fixed.jsonl"
IMAGE_SUBFOLDER = "images" 

fixed_count = 0

with open(INPUT_FILE, 'r', encoding='utf-8') as f_in, \
     open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
    
    for line in f_in:
        row = json.loads(line)
        
        if row.get("modality") == "image":
            # Extract the hash from the quote_id
            # Example quote_id: "train_L1_image_1" or a hash-based ID
            # If your quote_id looks like "0b85477..._image6", we split it:
            quote_id = row.get("quote_id", "")
            tag = row.get("image_tag", "") # e.g., "image6"
            
            # Logic: We need to find the document hash. 
            # In your data, the line_idx or a specific field usually maps to this.
            # Based on your example, the format is: {doc_hash}_{tag}.jpg
            
            # Since I don't have the hash mapping, we can try to find the file 
            # that ends with _{tag}.jpg in your directory if the hash isn't in the JSON.
            # BUT, usually the 'quote_id' contains the info we need.
            
            # Let's assume the hash is derived from the 'quote_id' logic.
            # REVISED LOGIC: If the image path is missing, we construct it.
            # If your files are named like '0b85477387a9d0cc33fca0f4becaa0e5_image6.jpg'
            
            # Check if quote_id has the hash:
            doc_hash = quote_id.split('_')[0] 
            reconstructed_path = f"{IMAGE_SUBFOLDER}/{doc_hash}_{tag}.jpg"
            
            row["image_path"] = reconstructed_path
            fixed_count += 1
            
        f_out.write(json.dumps(row) + "\n")

print(f"✅ Repair Complete! Fixed {fixed_count} entries.")
print(f"Example path created: {reconstructed_path}")