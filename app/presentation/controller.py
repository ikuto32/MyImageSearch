
import logging
import json
import math
import re
import binascii
from io import BytesIO
from flask import Flask, Response, request, make_response, send_file, send_from_directory, abort, jsonify
from PIL import Image as PILImage

import pathlib
import base64
import mimetypes

import numpy as np

from app.application.usecase import Usecase
from app.application.text_matching import SearchMatcher, SearchPatternError
from app.domain.errors import ResourceLimitError, SearchInputError
from app.domain.domain_object import ImageItem, ImageId, ModelId, ModelItem, ResultImageItemList, UploadImage, ResultImageItem, UploadText
from app.domain.search_query import InvalidSearchQuery, parse_search_query
from app.infrastructure.model_metadata import normalize_model_id
from app.logging_config import configure_logging
from app.presentation.download_store import DownloadStore, DownloadBusyError


# ============================================================
# 初期化処理

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 96 * 1024 * 1024
usecase: Usecase = None
logger = logging.getLogger(__name__)
download_store = DownloadStore()


DEFAULT_RESULT_SIZE = 2048
MAX_RESULT_SIZE = 8192
DEFAULT_IMAGE_ITEM_PAGE_SIZE = 60
MAX_IMAGE_ITEM_PAGE_SIZE = 240
MAX_IMAGE_SEARCH_IDS = 64
MAX_RATING_IDS = 8192
MAX_ZIP_IDS = 1024


class InputValidationError(ValueError):
    """Invalid API input, distinct from failures inside the use case."""


@app.errorhandler(InputValidationError)
@app.errorhandler(InvalidSearchQuery)
@app.errorhandler(SearchPatternError)
@app.errorhandler(SearchInputError)
def handle_input_validation_error(error):
    code = ("invalid_query" if isinstance(error, InvalidSearchQuery) else
            "invalid_pattern" if isinstance(error, SearchPatternError) else
            "invalid_model_or_image" if isinstance(error, SearchInputError) else "invalid_input")
    return {"error": str(error), "error_code": code}, 400


@app.errorhandler(413)
def handle_request_too_large(_error):
    return {"error": "リクエストが大きすぎます。ファイルや画像の選択数を減らしてください。", "error_code": "request_too_large"}, 413


@app.errorhandler(ResourceLimitError)
def handle_resource_limit(error):
    return {"error": str(error), "error_code": "resource_limit"}, 413


@app.errorhandler(DownloadBusyError)
def handle_download_busy(_error):
    return {"error": "他のZIP保存が進行中です。完了後に再試行してください。", "error_code": "download_busy"}, 429


@app.route("/downloads/prepare", methods=["POST"])
def prepare_download():
    params = get_request_params()
    ids = []

    def build():
        # Admit the operation before a filtered first-N catalog query can scan
        # a large database. Busy downloads must not start additional SQL work.
        ids.extend(download_id_parameters(params))
        return usecase.get_images_zip(ids)

    token, stream, size = download_store.prepare(build)
    return jsonify({
        "download_url": f"/downloads/{token}", "bytes": size,
        "requested_count": getattr(stream, "requested_count", len(ids)),
        "image_count": getattr(stream, "image_count", len(ids)),
        "skipped_count": getattr(stream, "skipped_count", 0),
    })


@app.route("/downloads/<token>", methods=["GET"])
def consume_download(token):
    stream = download_store.consume(token)
    if stream is None:
        return {"error": "保存リンクの有効期限が切れています。もう一度保存してください。", "error_code": "download_expired"}, 410
    try:
        stream.seek(0, 2)
        content_length = stream.tell()
        stream.seek(0)
        response = send_file(stream, mimetype="application/zip", as_attachment=True, download_name="images.zip")
    except BaseException:
        stream.close()
        raise
    response.headers["Cache-Control"] = "no-store"
    response.content_length = content_length
    response.call_on_close(stream.close)
    return response


@app.after_request
def add_response_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


def integer_parameter(value, name: str, minimum: int, maximum: int | None = None) -> int:
    # Query strings and older clients can supply integers as strings.
    if isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value):
        try:
            value = int(value)
        except ValueError:
            raise InputValidationError(f"{name} must be an integer") from None
    if type(value) is not int:
        raise InputValidationError(f"{name} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        limit = f"{minimum} to {maximum}" if maximum is not None else f"at least {minimum}"
        raise InputValidationError(f"{name} must be {limit}")
    return value


def get_result_size(params: dict | None, default: int = DEFAULT_RESULT_SIZE) -> int:
    """検索結果は既定2048件、最大8192件に制限する。"""
    return integer_parameter(
        (params or {}).get("result_size", default), "result_size", 1, MAX_RESULT_SIZE
    )


def start_app(in_usecase: Usecase, *, host: str = "0.0.0.0", port: int = 80):
    """エントリーポイント"""

    configure_logging()

    # ユースケースのDI
    global usecase
    usecase = in_usecase

    # text/javascript が text/plain になる問題の対策
    # https://bugs.python.org/issue43975
    mimetypes.add_type("text/javascript", ".js", True)

    np.set_printoptions(threshold=4096)

    # ログレベルをWARNING以上に設定することで、INFOレベルのログを非表示にします
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.WARNING)

    # Flask実行
    logger.info("Flask URL map: %s", app.url_map)
    app.run(debug=False, port=port, host=host)


# ============================================================
# エンドポイント

@app.route('/')
def index():
    """メインページを返す"""

    # ビューの指定をしている。
    pwd = pathlib.Path(__file__).parent
    with open(f'{pwd}/view/index.html', encoding="UTF-8") as f:
        text = f.read()

    return text


@app.route("/<path:target>")
def resource(target: str):
    """ファイルを返す"""

    # ビューの指定をしている。
    pwd = pathlib.Path(__file__).parent
    base_path = (pwd / "view").resolve()
    response = send_from_directory(str(base_path), target)
    content_type = mimetypes.guess_type(target)[0]
    if content_type is not None:
        response.headers.set('Content-Type', content_type)
    return response


# ------------------------------------------------------------


@app.route("/image_item")
def get_all_image_item():
    """ページングされた画像項目を返す"""

    page = integer_parameter(request.args.get("page", 0), "page", 0)
    page_size = integer_parameter(
        request.args.get("size", DEFAULT_IMAGE_ITEM_PAGE_SIZE),
        "size", 1, MAX_IMAGE_ITEM_PAGE_SIZE,
    )

    model_id = model_id_parameter(request.args) if "model_name" in request.args else None
    ratings = None
    if "ratings" in request.args:
        raw_ratings = request.args["ratings"]
        try:
            if len(raw_ratings) > 256:
                raise ValueError()
            ratings = json.loads(raw_ratings)
        except (ValueError, TypeError):
            raise InputValidationError("ratings must be a JSON array of supported image categories")
        if not isinstance(ratings, list):
            raise InputValidationError("ratings must be a JSON array of supported image categories")
        validate_image_ratings(ratings)
    total = matching = None
    if request.args.get("include_total") == "1":
        total = usecase.get_catalog_count(model_id)
        matching = usecase.get_catalog_count(model_id, ratings) if ratings is not None else total
    if isinstance(matching, int) and page * page_size >= matching:
        # A known-empty category or a thumb position past the end needs no
        # additional source-table scan to rediscover that there are no rows.
        items = []
    elif ratings is not None:
        items = usecase.get_image_items_by_page(page, page_size, model_id, ratings)
    else:
        items = (usecase.get_image_items_by_page(page, page_size, model_id)
                 if model_id is not None else usecase.get_image_items_by_page(page, page_size))
    response = from_image_item_list_to_json(items)
    if request.args.get("include_total") == "1":
        if total is not None:
            response.headers["X-Catalog-Total"] = str(total)
        if matching is not None:
            response.headers["X-Matching-Total"] = str(matching)
    return response


@app.route("/image_item/<id>")
def get_image_item(id: str):
    """画像IDから画像項目を返す"""

    try:
        item = usecase.get_image_item(ImageId(id))
    except (KeyError, ValueError):
        abort(404)
    return from_image_item_to_json(item)


@app.route("/image_meta/<id>", methods=["GET", "POST"])
def get_image_metadata(id: str):
    """画像のメタデータ（タグやratingなど）を返す"""

    model_id = get_requested_model_id()
    return jsonify(usecase.get_image_metadata(model_id, ImageId(id)))


@app.route("/image_ratings", methods=["GET", "POST"])
def get_image_ratings():
    """指定された画像IDに限定してrating一覧を返す"""

    model_id = get_requested_model_id()
    image_ids = get_requested_image_ids()
    ratings: dict[ImageId, str] = usecase.get_rating_list(model_id, image_ids)
    rating_dict = {image_id.id: rating for image_id, rating in ratings.items()}
    return jsonify(rating_dict)


def get_request_params() -> dict:
    """query string またはJSON bodyのparamsからAPIパラメータを取り出す"""

    if request.method != "POST":
        return request.args
    try:
        json_obj = request.get_json(silent=True)
    except RecursionError:
        raise InputValidationError("Request JSON is nested too deeply") from None
    if not isinstance(json_obj, dict):
        raise InputValidationError("Request body must be a JSON object")
    body_params = json_obj.get("params", json_obj)
    if not isinstance(body_params, dict):
        raise InputValidationError("params must be a JSON object")
    return body_params


def string_parameter(value, name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise InputValidationError(f"{name} must be a {'string' if allow_empty else 'non-empty string'}")
    return value


def number_parameter(value, name: str, minimum: float, maximum: float) -> float:
    if type(value) not in (int, float):
        raise InputValidationError(f"{name} must be a finite number")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite or not minimum <= value <= maximum:
        raise InputValidationError(f"{name} must be a finite number from {minimum} to {maximum}")
    return float(value)


def boolean_parameter(value, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if value in ("true", "false"):
        return value == "true"
    raise InputValidationError(f"{name} must be true or false")


def model_id_parameter(params: dict) -> ModelId:
    model_name = string_parameter(params.get("model_name"), "model_name")
    pretrained = params.get("pretrained")
    if pretrained is None:
        pretrained = ""
    return make_model_id(
        model_name, string_parameter(pretrained, "pretrained", allow_empty=True)
    )


def search_options(params: dict) -> tuple:
    """Validate shared search settings before starting any index/model work."""
    beta = number_parameter(params.get("aesthetic_quality_beta", 0), "aesthetic_quality_beta", -1, 1)
    quality_range = params.get("aesthetic_quality_range", [0, 10])
    if not isinstance(quality_range, list) or len(quality_range) != 2:
        raise InputValidationError("aesthetic_quality_range must contain two numbers")
    low, high = (
        number_parameter(value, "aesthetic_quality_range", 0, 10)
        for value in quality_range
    )
    if low > high:
        raise InputValidationError("aesthetic_quality_range minimum must not exceed maximum")
    model_name = params.get("aesthetic_model_name")
    if model_name is None:
        model_name = "original"
    model_name = string_parameter(model_name, "aesthetic_model_name")
    if model_name not in ("original", "pony"):
        raise InputValidationError("aesthetic_model_name must be original or pony")
    return beta, low, high, model_name, get_result_size(params)


def text_parameters(params: dict) -> tuple[str, bool]:
    text = string_parameter(params.get("text"), "text")
    is_regexp = boolean_parameter(params.get("is_regexp", False), "is_regexp")
    SearchMatcher(text, is_regexp)
    return text, is_regexp


def image_ids_parameter(value, name: str, *, allow_empty: bool = False, maximum: int = MAX_RATING_IDS) -> list[str]:
    if not isinstance(value, list) or (not allow_empty and not value):
        raise InputValidationError(f"{name} must be a {'list' if allow_empty else 'non-empty list'} of image IDs")
    if len(value) > maximum:
        raise InputValidationError(f"{name} must contain at most {maximum} image IDs")
    return [string_parameter(item, name) for item in value]


def query_parameter(params: dict) -> str:
    return parse_search_query(params.get("search_query"), model_id_parameter(params)).to_text()


def make_model_id(model_name: str, pretrained: str | None = "") -> ModelId:
    return normalize_model_id(ModelId(model_name, pretrained or ""))


def get_requested_model_id() -> ModelId:
    """リクエストパラメータから検索モデルIDを取り出す"""

    params = get_request_params()
    return model_id_parameter(params)


def get_requested_image_ids() -> list[ImageId] | None:
    """rating取得リクエストから対象画像IDリストを取り出す"""

    params = get_request_params()
    if request.method == "POST":
        id_text_list = params.get("id", params.get("ids"))
    else:
        id_text_list = request.args.getlist("id") or request.args.getlist("ids")
        if not id_text_list:
            ids_text = params.get("ids")
            id_text_list = ids_text.split(",") if ids_text else None

    if id_text_list is None:
        raise InputValidationError("ids is required; specify the image IDs to retrieve")
    if isinstance(id_text_list, str):
        id_text_list = [id_text_list]

    return [ImageId(item_id) for item_id in image_ids_parameter(id_text_list, "ids", allow_empty=True)]


@app.route("/image/<id>/small")
def get_small_image(id: str):
    """画像IDから縮小画像を返す"""

    try:
        img = usecase.get_small_image(ImageId(id))
    except ValueError:
        abort(404)
    response = make_response(img.binary)
    response.headers.set('Content-Type', img.content_type)
    return response


@app.route("/image/<id>/original")
def get_original_image(id: str):
    """画像IDからオリジナル画像を返す"""

    try:
        img = usecase.get_image(ImageId(id))
    except ValueError:
        abort(404)
    response = make_response(img.binary)
    response.headers.set('Content-Type', img.content_type)
    return response


# ------------------------------------------------------------


@app.route("/search/text", methods=["POST"])
def search_text():
    """文字列から検索して、結果を返す"""
    params = get_request_params()
    model_id = model_id_parameter(params)
    text = string_parameter(params.get("text"), "text")
    result = usecase.search_text(model_id, UploadText(text), *search_options(params))
    return from_result_to_json(result)


@app.route("/search/image", methods=["POST"])
def search_image():
    """画像から検索して、結果を返す"""
    params = get_request_params()
    model_id = model_id_parameter(params)
    id_list = [ImageId(item) for item in image_ids_parameter(params.get("id"), "id", maximum=MAX_IMAGE_SEARCH_IDS)]
    result = usecase.search_image(model_id, id_list, *search_options(params))
    return from_result_to_json(result)


@app.route("/search/name", methods=["POST"])
def search_name():
    """名前から検索して、結果を返す"""
    params = get_request_params()
    model_id = model_id_parameter(params)
    text, is_regexp = text_parameters(params)
    result = usecase.search_name(model_id, UploadText(text), is_regexp, *search_options(params))
    return from_result_to_json(result)


@app.route("/search/uploadimage", methods=["POST"])
def search_upload_image():
    """アップロードされた画像から検索して、結果を返す。"""
    params = get_request_params()
    model_id = model_id_parameter(params)
    base64_text = string_parameter(params.get("base64"), "base64")
    content_type = string_parameter(params.get("content_type", "image/png"), "content_type")
    if not content_type.startswith("image/"):
        raise InputValidationError("content_type must be an image MIME type")
    if "," in base64_text:
        base64_text = base64_text.split(",", 1)[1]
    try:
        binary = base64.b64decode(base64_text, validate=True)
    except (binascii.Error, ValueError):
        raise InputValidationError("base64 must contain a valid encoded image") from None
    if len(binary) > 64 * 1024 * 1024:
        raise ResourceLimitError("アップロード画像は64MiB以内で指定してください。")
    image = UploadImage(binary, content_type)
    try:
        with PILImage.open(BytesIO(binary)) as uploaded_image:
            if uploaded_image.width * uploaded_image.height > 64_000_000:
                raise ResourceLimitError("アップロード画像は6400万画素以内で指定してください。")
            uploaded_image.verify()
    except PILImage.DecompressionBombError:
        raise ResourceLimitError("アップロード画像は6400万画素以内で指定してください。") from None
    except (OSError, ValueError, SyntaxError):
        raise InputValidationError("base64 must contain a readable image") from None
    result_size = get_result_size(params)
    result = usecase.search_upload_image(model_id, image, result_size)
    return from_result_to_json(result)


@app.route("/search/random", methods=["POST"])
def search_random():
    """乱数から検索して、結果を返す"""
    params = get_request_params()
    model_id = model_id_parameter(params)
    result = usecase.search_random(model_id, *search_options(params))
    return from_result_to_json(result)


@app.route("/search/query", methods=["POST"])
def search_query():
    """クエリから検索して、結果を返す"""
    params = get_request_params()
    model_id = model_id_parameter(params)
    query = query_parameter(params)
    result = usecase.search_query(model_id, query, *search_options(params))
    return from_result_to_json(result)


@app.route("/search/queryaddtext", methods=["POST"])
def add_text_features():
    """クエリにテキストの特徴を足してから検索して、結果を返す"""
    params = get_request_params()
    model_id = model_id_parameter(params)
    text = string_parameter(params.get("text"), "text")
    query = query_parameter(params)
    strength = number_parameter(params.get("features_strength", 1), "features_strength", -2, 2)
    result = usecase.add_text_features(model_id, UploadText(text), query, strength, *search_options(params))
    return from_result_to_json(result)


@app.route("/search/tags", methods=["POST"])
def search_tags():
    """タグから検索して、結果を返す"""
    params = get_request_params()
    model_id = model_id_parameter(params)
    text, is_regexp = text_parameters(params)
    result = usecase.search_tags(model_id, UploadText(text), is_regexp, *search_options(params))
    return from_result_to_json(result)


@app.route("/search/style_cluster", methods=["POST"])
def search_style_cluster():
    """style_cluster から検索して、結果を返す"""
    params = get_request_params()
    model_id = model_id_parameter(params)
    text, is_regexp = text_parameters(params)
    result = usecase.search_style_cluster(model_id, UploadText(text), is_regexp, *search_options(params))
    return from_result_to_json(result)

@app.route("/download_images_zip", methods=["POST"])
def download_images_zip():
    params = get_request_params()
    id_list = download_id_parameters(params)

    zip_buffer = usecase.get_images_zip(id_list)

    try:
        response = send_file(
            zip_buffer,
            mimetype='application/zip',
            as_attachment=True,
            download_name='images.zip'
        )
    except Exception:
        zip_buffer.close()
        raise
    response.call_on_close(zip_buffer.close)
    for attribute, header in (("requested_count", "X-Archive-Requested"), ("image_count", "X-Archive-Images"), ("skipped_count", "X-Archive-Skipped")):
        count = getattr(zip_buffer, attribute, None)
        if isinstance(count, int):
            response.headers[header] = str(count)
    return response


def download_id_parameters(params: dict) -> list[str]:
    """Accept selected IDs or a first-N catalog request, never both."""
    if "first" not in params:
        return image_ids_parameter(params.get("ids"), "ids", maximum=MAX_ZIP_IDS)
    if "ids" in params:
        raise InputValidationError("Specify either ids or first, not both")
    limit = integer_parameter(params["first"], "first", 1, MAX_ZIP_IDS)
    model_id = model_id_parameter(params)
    ratings = params.get("ratings")
    validate_image_ratings(ratings)
    return usecase.get_download_ids(model_id, limit, ratings)


def validate_image_ratings(ratings):
    allowed = {"general", "sensitive", "questionable", "explicit", "unclassified"}
    if ratings is not None and (
        not isinstance(ratings, list) or len(ratings) > 5
        or any(not isinstance(value, str) or value not in allowed for value in ratings)
    ):
        raise InputValidationError("ratings must be a list of supported image categories")
# ============================================================


def from_image_item_to_json(value: ImageItem) -> Response:

    obj = {"id": value.id.id, "name": value.display_name.name, "tags": value.tags.tags}
    return jsonify(obj)


def from_image_item_list_to_json(value: list[ImageItem]) -> Response:

    objs = map(lambda r: {"id": r.id.id, "name": r.display_name.name, "tags": r.tags.tags, "rating": r.rating}, value)
    objs = list(objs)
    return jsonify(objs)


def from_result_to_json(value: ResultImageItemList) -> Response:

    objs = map(lambda r: {
        "item": {
            "id": str(r.item.id.id),
            "name": str(r.item.display_name.name),
            "tags": str(r.item.tags.tags)
        },
        "score": float(r.score.score)
    },
        value.list
    )

    return jsonify({
        "list": list(objs),
        "search_query": value.search_query
    })


def from_result_list_to_json(value: list[ResultImageItem]) -> Response:

    objs = map(lambda r: {
        "item": {
            "id": str(r.item.id.id),
            "name": str(r.item.display_name.name),
            "tags": str(r.item.tags.tags)
        },
        "score": float(r.score.score)
    },
        value
    )

    return jsonify(list(objs))


def from_model_id_to_json(value: list[ModelItem]) -> Response:

    objs = map(lambda r: {"model_name": r.id.model_name,
               "pretrained": r.id.pretrained}, value)
    objs = list(objs)
    return jsonify(objs)


# ============================================================

@app.route("/model_item")
def get_all_model_item():
    return from_model_id_to_json(usecase.get_all_model())
