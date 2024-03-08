import sys
import os
import json
import re
import base64

REGEX_PATTERN = r'!\[(.*?)\]\((?:\w+?:)?(.*?)\)'
ATTACHMENTS_KEY = 'attachments'
ATTACHMENT_QUALIFIER = 'attachment'
IMAGE_KEY = 'image/png'

image_directory = sys.argv[1]
ipynb_filename = sys.argv[2]
output_filename = sys.argv[3]

def image_to_base64(image_filename):
    with open(image_filename, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_image


with open(ipynb_filename, "r") as ipynb_file:
    ipynb_dict = json.load(ipynb_file)

for i, cell in enumerate(ipynb_dict['cells']):
    if cell['cell_type'] == 'markdown':
        for j, line in enumerate(cell['source']):
            matches = re.findall(REGEX_PATTERN, line)
            if len(matches) > 0:
                for description, image_filename in matches:
                    print(description, image_filename)

                    substitute_by = f"![{description}]({ATTACHMENT_QUALIFIER}:{image_filename})"
                    new_line = re.sub(REGEX_PATTERN, substitute_by, line)
                    ipynb_dict['cells'][i]['source'][j] = new_line
                    
                    image_path = os.path.join(image_directory, image_filename)
                    encoded_image = image_to_base64(image_path)
                    attachment_dict = {image_filename: {IMAGE_KEY: encoded_image}}
                    ipynb_dict['cells'][i][ATTACHMENTS_KEY] = attachment_dict


with open(output_filename, 'w') as output_file:
    json.dump(
        ipynb_dict,
        output_file,
        sort_keys=True,
        indent=2,
        separators=(',', ': ')
    )
