import os
import io
import json

import responder

from color_extraction import load_img, ColorExtractor


env = os.environ
DEBUG = env['DEBUG'] in ['1', 'True', 'true']
ALGO = env['ALGO']

cur_dir = os.path.dirname(__file__)
with open(os.path.join(cur_dir, 'color_extractor.json')) as fp:
    CONFIG = json.load(fp)

api = responder.API(debug=DEBUG)
color_extractor = ColorExtractor(
    algo=ALGO, **CONFIG[ALGO]
)


def extract_colors(bytes_io):
    img = load_img(bytes_io)
    return color_extractor.extract(img)


@api.route("/")
async def extract(req, resp):
    body = await req.content
    data = extract_colors(io.BytesIO(body))
    resp.media = dict(data=data)


if __name__ == "__main__":
    api.run()