import requests
import os

def test_model():
    """Test function untuk memastikan model berjalan dengan baik"""
    try:
        # Ganti dengan URL Railway nanti
        base_url = os.getenv('RAILWAY_STATIC_URL', 'http://localhost:8080')
        
        # Health check
        health_url = f"{base_url}/v1/models/drug_classification"
        response = requests.get(health_url)
        
        if response.status_code == 200:
            print("Model is running successfully!")
            print(f"Model status: {response.json()}")
            return True
        else:
            print(f"Model health check failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"Error testing model: {e}")
        return False

if __name__ == "__main__":
    test_model()