import unittest
from unittest.mock import Mock, patch
from app.presentation import controller
from app.domain.domain_object import ImageId, ImageItem, ImageName, ModelId


class CatalogModelSelectionTests(unittest.TestCase):
    def test_selected_model_and_total_match_page_metadata(self):
        usecase = Mock()
        usecase.get_image_items_by_page.return_value = [ImageItem(ImageId('other'), ImageName('other.png'))]
        usecase.get_catalog_count.return_value = 1234
        with patch.object(controller, 'usecase', usecase):
            response = controller.app.test_client().get('/image_item?page=2&size=60&model_name=other&pretrained=weights&include_total=1')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()[0]['id'], 'other')
        self.assertEqual(response.headers['X-Catalog-Total'], '1234')
        usecase.get_image_items_by_page.assert_called_once_with(2, 60, ModelId('other', 'weights'))
        usecase.get_catalog_count.assert_called_once_with(ModelId('other', 'weights'))

    def test_total_is_not_recounted_for_plain_page_requests(self):
        usecase = Mock()
        usecase.get_image_items_by_page.return_value = []
        with patch.object(controller, 'usecase', usecase):
            response = controller.app.test_client().get('/image_item?page=0&size=60')
        self.assertEqual(response.status_code, 200)
        usecase.get_catalog_count.assert_not_called()


if __name__ == '__main__':
    unittest.main()
