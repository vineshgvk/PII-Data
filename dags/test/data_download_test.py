# import unittest
# import os
# import pickle
# import sys

# # Append the path of the 'dags' directory to sys.path
# current_dir = os.path.dirname(__file__)
# parent_dir = os.path.abspath(os.path.join(current_dir, '..', 'src'))
# sys.path.insert(0, parent_dir)

# from data_download import load_data_from_gcp_and_save_as_json, PROJECT_DIR
# class TestDataDownload(unittest.TestCase):

#     def setUp(self):
#         """Setup before each test."""
#         self.pickle_path = os.path.join(PROJECT_DIR, "data", "processed", "test.json")
        
#     def test_load_data_from_gcp_and_save_as_json(self):
#         """Test the load_data_from_gcp_and_save_as_json function."""
#         # Assuming the function creates a pickle file at the specified location.
#         # This will not actually call the Google Cloud function but ensures
#         # the file path logic and local file handling are correct.
        
#         # Call the function with the test path
#         load_data_from_gcp_and_save_as_json(self.pickle_path)
        
#         # Check if the file was created
#         self.assertTrue(os.path.exists(self.pickle_path))
        
#         # Optionally, load the pickle file and assert its contents
#         # This part is commented out because we are not interacting with GCP in this test
#         # with open(self.pickle_path, 'rb') as f:
#         #     data = pickle.load(f)
#         # self.assertIsInstance(data, list)  # Example assertion

#     def tearDown(self):
#         """Clean up after each test."""
#         # Remove the created test pickle file
#         if os.path.exists(self.pickle_path):
#             os.remove(self.pickle_path)

# if __name__ == '__main__':
#     unittest.main()
