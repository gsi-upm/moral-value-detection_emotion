import unittest
from fastapi.testclient import TestClient
from main import app  

#!pip install httpx

class TestFastAPI(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    def test_predict_moral_model(self):
        """Test 'moral_model'"""
        data = {
            "text": "The government should protect its citizens and maintain law and order.",
            "model_name": "moral_model"
        }
        response = self.client.post("/predict", json=data)
        
        self.assertEqual(response.status_code, 200)
        self.assertIn("Predicted_Moral", response.json())


    def test_predict_moralpolarity_model(self):
        """Test 'moralpolarity_model'"""
        data = {
            "text": "The government should protect its citizens and maintain law and order.",
            "model_name": "moralpolarity_model"
        }
        response = self.client.post("/predict", json=data)
        
        self.assertEqual(response.status_code, 200)  
        self.assertIn("Predicted_Moral_Polarity", response.json())


    def test_predict_multimoralpolarity_model(self):
        """Test 'multimoralpolarity_model'"""
        data = {
            "text": "The government should protect its citizens and maintain law and order.",
            "model_name": "multimoralpolarity_model"
        }
        response = self.client.post("/predict", json=data)
        
        self.assertEqual(response.status_code, 200)
        self.assertIn("Predicted_Moral", response.json())

    def test_predict_multimoral_model(self):
        """Test 'multimoral_model'"""
        data = {
            "text": "The government should protect its citizens and maintain law and order.",
            "model_name": "multimoral_model"
        }
        response = self.client.post("/predict", json=data)
        
        self.assertEqual(response.status_code, 200)
        self.assertIn("Predicted_Moral_Trait", response.json())

    def test_invalid_model(self):
        data = {
            "text": "The government should protect its citizens and maintain law and order.",
            "model_name": "non_existing_model"
        }
        response = self.client.post("/predict", json=data)
        
        self.assertEqual(response.status_code, 400) 
        self.assertIn("error", response.json())  


if __name__ == "__main__":
    unittest.main()
