import requests
import json
import datetime
import time
import os
import statistics
from dotenv import dotenv_values
from typing import Dict, List, Any, Optional
import pandas as pd 
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

PEST_DATA = pd.read_csv("Datasets/pest_.csv")
PEST_DATA.head()

# Create a mapping dictionary for rain values to percentages
rain_mapping = {
    'Low': 20,       # Low rain -> 20% chance
    'Moderate': 50,  # Moderate rain -> 50% chance
    'High': 80       # High rain -> 80% chance
}

# Create a new column with numerical percentage values
PEST_DATA['rain_percentage'] = PEST_DATA['rain'].map(rain_mapping)

# Display the first few rows to verify the conversion
PEST_DATA[['plant', 'rain', 'rain_percentage']].head()

class WeatherAnalyzer:
    def __init__(self, api_key: str, city: str, data_dir: str = "weather_data"):
        """
        Initialize the Weather Analyzer with API key and city
        
        Parameters:
        - api_key: OpenWeather API key
        - city: City name to analyze weather for
        - data_dir: Directory to store historical weather data
        """
        self.api_key = api_key
        self.city = city
        self.data_dir = data_dir
        self.current_data = None
        self.forecast_data = None
        self.historical_data = []
        
        # Create data directory if it doesn't exist
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            
        # Load existing historical data if available
        self._load_historical_data()
    
    def calculate_rain_chance(self, weather_data: Dict[str, Any]) -> int:
        """
        Calculate chance of rain based on weather parameters
        
        Parameters:
        - weather_data: Weather data dictionary
        
        Returns:
        - Integer from 0-100 representing rain probability
        """
        rain_chance = 0
        
        # Check weather condition
        weather_type = weather_data.get("weather_main", "").lower()
        if "rain" in weather_type or "drizzle" in weather_type:
            rain_chance += 70
        elif "shower" in weather_type:
            rain_chance += 60
        elif "thunderstorm" in weather_type:
            rain_chance += 80
        elif "clouds" in weather_type:
            rain_chance += 30
        
        # Factor in humidity (higher humidity increases rain chance)
        humidity = weather_data.get("humidity", 0)
        rain_chance += min(humidity / 5, 20)  # Up to 20 points for humidity
        
        # Factor in cloud coverage
        clouds = weather_data.get("clouds", 0)
        rain_chance += clouds / 5  # Up to 20 points for full cloud coverage
        
        # Pressure factor (lower pressure often means higher rain chance)
        # Standard pressure is around 1013 hPa
        pressure = weather_data.get("pressure", 1013)
        if pressure < 1005:
            rain_chance += 10
        elif pressure < 1000:
            rain_chance += 15
        
        # Ensure we stay within 0-100 range
        return max(0, min(100, int(rain_chance)))
    
    def fetch_current_weather(self) -> Dict[str, Any]:
        """Fetch current weather data from OpenWeather API"""
        url = f"https://api.openweathermap.org/data/2.5/weather?q={self.city}&appid={self.api_key}&units=metric"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            self.current_data = response.json()
            
            # Clean and format the data
            cleaned_data = {
                "timestamp": datetime.datetime.now().isoformat(),
                "fetch_time": int(time.time()),
                "temp": self.current_data["main"]["temp"],
                "feels_like": self.current_data["main"]["feels_like"],
                "temp_min": self.current_data["main"]["temp_min"],
                "temp_max": self.current_data["main"]["temp_max"],
                "pressure": self.current_data["main"]["pressure"],
                "humidity": self.current_data["main"]["humidity"],
                "wind_speed": self.current_data["wind"]["speed"],
                "wind_direction": self.current_data["wind"]["deg"],
                "weather_main": self.current_data["weather"][0]["main"],
                "weather_description": self.current_data["weather"][0]["description"],
                "clouds": self.current_data["clouds"]["all"],
                "city": self.city
            }
            
            # Calculate and add rain chance
            cleaned_data["rain_chance"] = self.calculate_rain_chance(cleaned_data)
            
            # Save this data to historical records
            self._save_weather_data(cleaned_data)
            
            return cleaned_data
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching current weather: {e}")
            return {}
    
    def fetch_forecast(self, days: int = 5) -> Dict[str, Any]:
        """Fetch weather forecast data from OpenWeather API"""
        url = f"https://api.openweathermap.org/data/2.5/forecast?q={self.city}&appid={self.api_key}&units=metric"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            self.forecast_data = response.json()
            
            # Clean and format the forecast data
            cleaned_forecast = {
                "timestamp": datetime.datetime.now().isoformat(),
                "city": self.city,
                "forecast": []
            }
            
            for item in self.forecast_data["list"]:
                forecast_item = {
                    "dt": item["dt"],
                    "dt_txt": item["dt_txt"],
                    "temp": item["main"]["temp"],
                    "feels_like": item["main"]["feels_like"],
                    "temp_min": item["main"]["temp_min"],
                    "temp_max": item["main"]["temp_max"],
                    "pressure": item["main"]["pressure"],
                    "humidity": item["main"]["humidity"],
                    "weather_main": item["weather"][0]["main"],
                    "weather_description": item["weather"][0]["description"],
                    "clouds": item["clouds"]["all"],
                    "wind_speed": item["wind"]["speed"],
                    "wind_direction": item["wind"]["deg"]
                }
                
                # Calculate and add rain chance
                forecast_item["rain_chance"] = self.calculate_rain_chance(forecast_item)
                
                cleaned_forecast["forecast"].append(forecast_item)
            
            # Save only the first X days worth of forecast data
            forecasts_per_day = 8  # OpenWeather provides 8 forecasts per day (every 3 hours)
            limit = days * forecasts_per_day
            cleaned_forecast["forecast"] = cleaned_forecast["forecast"][:limit]
            
            return cleaned_forecast
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching forecast: {e}")
            return {}
            
    def _save_weather_data(self, data: Dict[str, Any]) -> None:
        """Save weather data to historical records"""
        # Add to in-memory historical data
        self.historical_data.append(data)
        
        # Save to file
        filename = f"{self.data_dir}/{self.city.lower()}_history.json"
        
        try:
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    existing_data = json.load(f)
            else:
                existing_data = []
                
            existing_data.append(data)
            
            with open(filename, 'w') as f:
                json.dump(existing_data, f, indent=2)
                
        except Exception as e:
            print(f"Error saving historical data: {e}")
    
    def _load_historical_data(self) -> None:
        """Load historical weather data from file"""
        filename = f"{self.data_dir}/{self.city.lower()}_history.json"
        
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as f:
                    self.historical_data = json.load(f)
            except Exception as e:
                print(f"Error loading historical data: {e}")
                self.historical_data = []
        else:
            self.historical_data = []
    
    def get_historical_data(self, days: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get historical weather data
        
        Parameters:
        - days: Number of days back to retrieve data (None for all data)
        
        Returns:
        - List of historical weather data points
        """
        if not days:
            return self.historical_data
        
        # Calculate timestamp for 'days' ago
        cutoff_time = int(time.time()) - (days * 24 * 60 * 60)
        
        return [data for data in self.historical_data if data.get("fetch_time", 0) >= cutoff_time]
    
    def generate_analysis_report(self, days_historical: int = 7, include_forecast: bool = True) -> Dict[str, Any]:
        """
        Generate comprehensive weather analysis report
        
        Parameters:
        - days_historical: Days of historical data to include in analysis
        - include_forecast: Whether to include forecast data
        
        Returns:
        - Dictionary containing analysis results
        """
        # Get current weather if we don't have it
        if not self.current_data:
            self.fetch_current_weather()
        
        # Get forecast if requested and we don't have it
        if include_forecast and not self.forecast_data:
            self.fetch_forecast()
        
        # Get historical data
        historical = self.get_historical_data(days_historical)
        
        # Initialize report dictionary
        report = {
            "city": self.city,
            "report_generated": datetime.datetime.now().isoformat(),
            "current_weather": self.fetch_current_weather() if self.current_data else {},
            "historical_analysis": {},
            "forecast_summary": {}
        }
        
        # Add historical analysis if we have data
        if historical:
            # Temperature analysis
            temps = [item["temp"] for item in historical if "temp" in item]
            if temps:
                report["historical_analysis"]["temperature"] = {
                    "average": round(statistics.mean(temps), 2),
                    "median": round(statistics.median(temps), 2),
                    "min": round(min(temps), 2),
                    "max": round(max(temps), 2),
                    "range": round(max(temps) - min(temps), 2),
                    "days_analyzed": days_historical
                }
            
            # Humidity analysis
            humidity = [item["humidity"] for item in historical if "humidity" in item]
            if humidity:
                report["historical_analysis"]["humidity"] = {
                    "average": round(statistics.mean(humidity), 2),
                    "median": round(statistics.median(humidity), 2),
                    "min": min(humidity),
                    "max": max(humidity)
                }
            
            # Wind analysis
            wind_speeds = [item["wind_speed"] for item in historical if "wind_speed" in item]
            if wind_speeds:
                report["historical_analysis"]["wind"] = {
                    "average_speed": round(statistics.mean(wind_speeds), 2),
                    "max_speed": round(max(wind_speeds), 2)
                }
            
            # Weather conditions summary
            weather_counts = {}
            for item in historical:
                if "weather_main" in item:
                    weather_type = item["weather_main"]
                    weather_counts[weather_type] = weather_counts.get(weather_type, 0) + 1
            
            if weather_counts:
                total = sum(weather_counts.values())
                weather_summary = {k: {"count": v, "percentage": round((v / total) * 100, 2)} 
                                  for k, v in weather_counts.items()}
                report["historical_analysis"]["weather_conditions"] = weather_summary
                
                # Determine predominant weather
                report["historical_analysis"]["predominant_weather"] = max(weather_counts, key=weather_counts.get)
        
        # Add forecast summary if requested
        if include_forecast and self.forecast_data:
            forecast_items = self.forecast_data["list"] if "list" in self.forecast_data else []
            
            if forecast_items:
                forecast_temps = [item["main"]["temp"] for item in forecast_items]
                forecast_weather = {}
                
                for item in forecast_items:
                    weather_type = item["weather"][0]["main"]
                    forecast_weather[weather_type] = forecast_weather.get(weather_type, 0) + 1
                
                report["forecast_summary"] = {
                    "avg_temp": round(statistics.mean(forecast_temps), 2),
                    "min_temp": round(min(forecast_temps), 2),
                    "max_temp": round(max(forecast_temps), 2),
                    "predominant_weather": max(forecast_weather, key=forecast_weather.get),
                    "forecast_hours": len(forecast_items)
                }
                
                # Add daily forecast summaries
                daily_forecasts = {}
                for item in forecast_items:
                    date = item["dt_txt"].split(" ")[0]
                    if date not in daily_forecasts:
                        daily_forecasts[date] = {
                            "temps": [],
                            "weather_types": []
                        }
                    
                    daily_forecasts[date]["temps"].append(item["main"]["temp"])
                    daily_forecasts[date]["weather_types"].append(item["weather"][0]["main"])
                
                report["forecast_summary"]["daily"] = {}
                for date, data in daily_forecasts.items():
                    # Count most common weather condition
                    weather_count = {}
                    for w in data["weather_types"]:
                        weather_count[w] = weather_count.get(w, 0) + 1
                    
                    most_common = max(weather_count, key=weather_count.get)
                    
                    report["forecast_summary"]["daily"][date] = {
                        "avg_temp": round(statistics.mean(data["temps"]), 2),
                        "min_temp": round(min(data["temps"]), 2),
                        "max_temp": round(max(data["temps"]), 2),
                        "predominant_weather": most_common
                    }
        
        # Add trend analysis if we have enough historical data and forecast
        if historical and include_forecast and self.forecast_data:
            if len(historical) >= 2:
                recent_temps = [item["temp"] for item in sorted(historical, key=lambda x: x.get("fetch_time", 0))[-3:]]
                recent_avg = statistics.mean(recent_temps)
                
                first_forecast_temp = self.forecast_data["list"][0]["main"]["temp"] if "list" in self.forecast_data and self.forecast_data["list"] else None
                
                if first_forecast_temp is not None:
                    report["trend_analysis"] = {
                        "recent_temp_trend": round(recent_temps[-1] - recent_temps[0], 2),
                        "forecast_vs_recent": round(first_forecast_temp - recent_avg, 2)
                    }
                    
                    # Provide a simple trend interpretation
                    if first_forecast_temp > recent_avg:
                        report["trend_analysis"]["temperature_outlook"] = "warming"
                    elif first_forecast_temp < recent_avg:
                        report["trend_analysis"]["temperature_outlook"] = "cooling"
                    else:
                        report["trend_analysis"]["temperature_outlook"] = "stable"
        
        return report