import logging
import mimetypes
import pathlib

from flask import Flask, Response, jsonify, make_response, request, send_file, send_from_directory

from app.application.text_matching import SearchPatternError
from app.application.usecase import MetadataField, Usecase
from app.domain import limits
from app.domain.domain_object import Image, ImageId, ImageItem, SearchResults
from app.domain.errors import ImageUnavailableError, ResourceLimitError, SearchInputError
from app.domain.search_query import InvalidSearchQuery
from app.logging_config import configure_logging
from app.presentation.download_store import DownloadBusyError, DownloadStore
from app.presentation.params import (
    DEFAULT_IMAGE_ITEM_PAGE_SIZE,
    MAX_IMAGE_ITEM_PAGE_SIZE,
    InputValidationError,
    image_ids_parameter,
    integer_parameter,
    json_params,
    model_id_parameter,
    number_parameter,
    pattern_parameters,
    ratings_parameter,
    ratings_query_parameter,
    search_options,
    search_query_parameter,
    text_parameter,
    uploaded_image_parameter,
)

logger = logging.getLogger(__name__)

VIEW_DIRECTORY = pathlib.Path(__file__).parent / "view"
# 64MiBの画像をbase64で送れる大きさ。
MAX_REQUEST_BYTES = 96 * 1024 * 1024
# 縮小画像はサーバー側でも再起動まで保持するため、ブラウザにも再利用させる。
THUMBNAIL_MAX_AGE_SECONDS = 3600

_ERROR_RESPONSES = {
    InputValidationError: (400, "invalid_input"),
    InvalidSearchQuery: (400, "invalid_query"),
    SearchPatternError: (400, "invalid_pattern"),
    SearchInputError: (400, "invalid_model_or_image"),
    ImageUnavailableError: (404, "image_unavailable"),
    ResourceLimitError: (413, "resource_limit"),
}


def _item_json(item: ImageItem) -> dict:
    # 一覧と検索結果には表示と絞り込みに使う項目だけを含める。タグなどは画像ごとに取得する。
    return {"id": item.id, "name": item.name, "rating": item.rating}


def _results_json(results: SearchResults) -> Response:
    return jsonify({
        "list": [{"item": _item_json(result.item), "score": result.score} for result in results.items],
        "search_query": results.search_query,
    })


def _image_response(image: Image) -> Response:
    response = make_response(image.binary)
    response.headers.set("Content-Type", image.content_type)
    return response


def create_app(usecase: Usecase, download_store: DownloadStore | None = None) -> Flask:
    """ユースケースを公開するFlaskアプリケーションを組み立てる"""

    # text/javascript が text/plain になる問題の対策
    # https://bugs.python.org/issue43975
    mimetypes.add_type("text/javascript", ".js", True)

    app = Flask(__name__, static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = MAX_REQUEST_BYTES
    downloads = download_store or DownloadStore()

    # ============================================================
    # エラーと共通ヘッダー

    def error_response(status: int, code: str):
        return lambda error: ({"error": str(error), "error_code": code}, status)

    for error_type, (status, code) in _ERROR_RESPONSES.items():
        app.register_error_handler(error_type, error_response(status, code))

    @app.errorhandler(413)
    def handle_request_too_large(_error):
        return {"error": "リクエストが大きすぎます。ファイルや画像の選択数を減らしてください。", "error_code": "request_too_large"}, 413

    @app.errorhandler(DownloadBusyError)
    def handle_download_busy(_error):
        return {"error": "他のZIP保存が進行中です。完了後に再試行してください。", "error_code": "download_busy"}, 429

    @app.after_request
    def add_response_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    # ============================================================
    # 画面

    @app.get("/")
    def index():
        """メインページを返す"""
        return send_from_directory(VIEW_DIRECTORY, "index.html")

    @app.get("/<path:target>")
    def resource(target: str):
        """画面のファイルを返す"""
        return send_from_directory(VIEW_DIRECTORY, target)

    # ============================================================
    # 一覧・画像

    @app.get("/model_item")
    def get_all_model_item():
        return jsonify([
            {"model_name": item.id.model_name, "pretrained": item.id.pretrained}
            for item in usecase.get_all_model()
        ])

    @app.get("/image_item")
    def get_image_items():
        """ページングされた画像項目を返す。include_total=1で件数をヘッダーに付ける。"""
        page = integer_parameter(request.args.get("page", 0), "page", 0)
        size = integer_parameter(
            request.args.get("size", DEFAULT_IMAGE_ITEM_PAGE_SIZE), "size", 1, MAX_IMAGE_ITEM_PAGE_SIZE,
        )
        model_id = model_id_parameter(request.args)
        ratings = ratings_query_parameter(request.args.get("ratings"))
        headers = {}
        matching = None
        if request.args.get("include_total") == "1":
            total = usecase.get_catalog_count(model_id)
            matching = total if ratings is None else usecase.get_catalog_count(model_id, ratings)
            headers = {"X-Catalog-Total": str(total), "X-Matching-Total": str(matching)}
        # A known-empty category or a thumb position past the end needs no
        # additional source-table scan to rediscover that there are no rows.
        if matching is not None and page * size >= matching:
            items = []
        else:
            items = usecase.get_image_items_by_page(model_id, page, size, ratings)
        response = jsonify([_item_json(item) for item in items])
        response.headers.update(headers)
        return response

    @app.get("/image_meta/<image_id>")
    def get_image_metadata(image_id: str):
        """画像のメタデータ（タグやratingなど）を返す"""
        item = usecase.get_image_metadata(model_id_parameter(request.args), ImageId(image_id))
        if item is None:
            raise ImageUnavailableError(f"画像IDが見つかりません: {image_id}")
        return jsonify({
            "tags": item.tags,
            "style_cluster": item.style_cluster,
            "rating": item.rating,
            "aesthetic_quality": item.aesthetic_quality,
        })

    @app.get("/image/<image_id>/small")
    def get_small_image(image_id: str):
        """画像IDから縮小画像を返す"""
        response = _image_response(usecase.get_small_image(ImageId(image_id)))
        response.cache_control.private = True
        response.cache_control.max_age = THUMBNAIL_MAX_AGE_SECONDS
        return response

    @app.get("/image/<image_id>/original")
    def get_original_image(image_id: str):
        """画像IDからオリジナル画像を返す"""
        return _image_response(usecase.get_image(ImageId(image_id)))

    # ============================================================
    # 検索

    @app.post("/search/text")
    def search_text():
        """文字列から検索して、結果を返す"""
        params = json_params()
        return _results_json(usecase.search_text(
            model_id_parameter(params), text_parameter(params), search_options(params),
        ))

    @app.post("/search/image")
    def search_image():
        """登録済みの画像から検索して、結果を返す"""
        params = json_params()
        return _results_json(usecase.search_image(
            model_id_parameter(params),
            image_ids_parameter(params.get("id"), "id", limits.MAX_IMAGE_SEARCH_IDS),
            search_options(params),
        ))

    @app.post("/search/uploadimage")
    def search_upload_image():
        """アップロードされた画像から検索して、結果を返す"""
        params = json_params()
        return _results_json(usecase.search_upload_image(
            model_id_parameter(params), uploaded_image_parameter(params), search_options(params),
        ))

    @app.post("/search/random")
    def search_random():
        """乱数から検索して、結果を返す"""
        params = json_params()
        return _results_json(usecase.search_random(model_id_parameter(params), search_options(params)))

    @app.post("/search/query")
    def search_query():
        """保存クエリから検索して、結果を返す"""
        params = json_params()
        return _results_json(usecase.search_query(
            model_id_parameter(params), search_query_parameter(params), search_options(params),
        ))

    @app.post("/search/queryaddtext")
    def add_text_features():
        """保存クエリにテキストの特徴を足してから検索して、結果を返す"""
        params = json_params()
        return _results_json(usecase.add_text_features(
            model_id_parameter(params),
            text_parameter(params),
            search_query_parameter(params),
            number_parameter(params.get("features_strength", 1), "features_strength", -2, 2),
            search_options(params),
        ))

    def search_metadata(field: MetadataField):
        params = json_params()
        model_id = model_id_parameter(params)
        text, is_regexp = pattern_parameters(params)
        return _results_json(usecase.search_metadata(model_id, field, text, is_regexp, search_options(params)))

    for field in ("name", "tags", "style_cluster"):
        app.add_url_rule(
            f"/search/{field}", f"search_{field}", search_metadata,
            methods=["POST"], defaults={"field": field},
        )

    # ============================================================
    # 保存

    def download_ids(params: dict) -> list[ImageId]:
        """Accept selected IDs or a first-N catalog request, never both."""
        if "first" not in params:
            return image_ids_parameter(params.get("ids"), "ids", limits.MAX_DOWNLOAD_IMAGES)
        if "ids" in params:
            raise InputValidationError("Specify either ids or first, not both")
        limit = integer_parameter(params["first"], "first", 1, limits.MAX_DOWNLOAD_IMAGES)
        model_id = model_id_parameter(params)
        return usecase.get_download_ids(model_id, limit, ratings_parameter(params.get("ratings")))

    @app.post("/downloads/prepare")
    def prepare_download():
        """ZIPを作成し、一度だけ受け取れるURLを返す"""
        params = json_params()
        archive = None

        def build():
            nonlocal archive
            # Admit the operation before a filtered first-N catalog query can scan
            # a large database. Busy downloads must not start additional SQL work.
            archive = usecase.get_images_zip(download_ids(params))
            return archive.stream

        token, size = downloads.prepare(build)
        return jsonify({
            "download_url": f"/downloads/{token}",
            "bytes": size,
            "requested_count": archive.requested_count,
            "image_count": archive.image_count,
            "skipped_count": archive.skipped_count,
        })

    @app.get("/downloads/<token>")
    def consume_download(token: str):
        if request.method != "GET":
            # GETのルートはHEADも受け付ける。確認だけのリクエストで一度きりのリンクを使い切らない。
            return "", 405, {"Allow": "GET"}
        stream = downloads.consume(token)
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

    return app


def start_app(usecase: Usecase, *, host: str = "127.0.0.1", port: int = 80) -> None:
    """エントリーポイント"""

    configure_logging()
    # リクエストごとのアクセスログは表示しない。
    logging.getLogger("werkzeug").setLevel(logging.WARNING)

    app = create_app(usecase)
    logger.debug("Flask URL map: %s", app.url_map)
    app.run(debug=False, port=port, host=host)
