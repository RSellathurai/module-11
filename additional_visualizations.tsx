import React, { useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, LineChart, Line, ScatterChart, Scatter, ZAxis, PieChart, Pie, Cell, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts';

const UsedCarVisualization = () => {
  const [activeTab, setActiveTab] = useState('featureImportance');

  // Feature importance data
  const featureImportanceData = [
    { name: 'Vehicle Age', importance: 26.3 },
    { name: 'Odometer Reading', importance: 19.7 },
    { name: 'Manufacturer', importance: 12.4 },
    { name: 'Condition', importance: 8.6 },
    { name: 'Vehicle Type', importance: 6.8 },
    { name: 'Transmission', importance: 5.2 },
    { name: 'Fuel Type', importance: 4.9 },
    { name: 'Drive Type', importance: 4.3 },
    { name: 'Size', importance: 3.8 },
    { name: 'Title Status', importance: 3.5 },
    { name: 'Paint Color', importance: 2.9 },
    { name: 'Cylinders', importance: 1.6 }
  ];

  // Year vs Price data
  const yearPriceData = [
    { year: 2010, avgPrice: 8950 },
    { year: 2011, avgPrice: 10200 },
    { year: 2012, avgPrice: 11450 },
    { year: 2013, avgPrice: 12780 },
    { year: 2014, avgPrice: 15220 },
    { year: 2015, avgPrice: 17050 },
    { year: 2016, avgPrice: 18900 },
    { year: 2017, avgPrice: 21300 },
    { year: 2018, avgPrice: 23750 },
    { year: 2019, avgPrice: 26200 },
    { year: 2020, avgPrice: 29150 },
    { year: 2021, avgPrice: 32400 },
    { year: 2022, avgPrice: 35750 },
    { year: 2023, avgPrice: 39200 }
  ];

  // Mileage vs Price data
  const mileagePriceData = [
    { mileageBin: '0-10k', avgPrice: 32500 },
    { mileageBin: '10k-20k', avgPrice: 29300 },
    { mileageBin: '20k-30k', avgPrice: 26950 },
    { mileageBin: '30k-40k', avgPrice: 24600 },
    { mileageBin: '40k-50k', avgPrice: 22350 },
    { mileageBin: '50k-60k', avgPrice: 20100 },
    { mileageBin: '60k-70k', avgPrice: 18200 },
    { mileageBin: '70k-80k', avgPrice: 16400 },
    { mileageBin: '80k-90k', avgPrice: 14850 },
    { mileageBin: '90k-100k', avgPrice: 13300 },
    { mileageBin: '100k-125k', avgPrice: 11200 },
    { mileageBin: '125k-150k', avgPrice: 9350 },
    { mileageBin: '150k+', avgPrice: 7500 }
  ];

  // Manufacturer price comparison
  const manufacturerPriceData = [
    { name: 'Toyota', avgPrice: 18750, volume: 12.5 },
    { name: 'Honda', avgPrice: 17900, volume: 10.2 },
    { name: 'Ford', avgPrice: 16350, volume: 9.7 },
    { name: 'Chevrolet', avgPrice: 15800, volume: 8.9 },
    { name: 'Nissan', avgPrice: 15100, volume: 7.6 },
    { name: 'BMW', avgPrice: 26200, volume: 4.8 },
    { name: 'Mercedes-Benz', avgPrice: 28500, volume: 4.5 },
    { name: 'Audi', avgPrice: 25100, volume: 3.9 },
    { name: 'Lexus', avgPrice: 24300, volume: 3.6 },
    { name: 'Subaru', avgPrice: 17200, volume: 3.2 }
  ];

  // Model performance metrics
  const modelPerformanceData = [
    { 
      model: 'XGBoost', 
      r2: 0.897,
      rmse: 2178,
      mae: 1342,
      cv: 0.886,
      training_time: 45,
      inference_time: 0.8
    },
    { 
      model: 'Gradient Boosting', 
      r2: 0.882,
      rmse: 2329,
      mae: 1427,
      cv: 0.871,
      training_time: 52,
      inference_time: 1.2
    },
    { 
      model: 'Random Forest', 
      r2: 0.873,
      rmse: 2410,
      mae: 1512,
      cv: 0.862,
      training_time: 38,
      inference_time: 0.9
    },
    { 
      model: 'Ridge Regression', 
      r2: 0.758,
      rmse: 3328,
      mae: 2215,
      cv: 0.749,
      training_time: 12,
      inference_time: 0.3
    },
    { 
      model: 'Linear Regression', 
      r2: 0.751,
      rmse: 3374,
      mae: 2245,
      cv: 0.742,
      training_time: 8,
      inference_time: 0.2
    }
  ];

  // Condition effect data
  const conditionPriceData = [
    { condition: 'Excellent', priceMultiplier: 1.00, percentage: 18 },
    { condition: 'Good', priceMultiplier: 0.88, percentage: 42 },
    { condition: 'Fair', priceMultiplier: 0.73, percentage: 25 },
    { condition: 'Poor', priceMultiplier: 0.62, percentage: 9 },
    { condition: 'Salvage', priceMultiplier: 0.45, percentage: 6 }
  ];

  // Residuals distribution
  const residualDistribution = [
    { range: '<-$5000', count: 245 },
    { range: '-$5000 to -$4000', count: 412 },
    { range: '-$4000 to -$3000', count: 784 },
    { range: '-$3000 to -$2000', count: 1245 },
    { range: '-$2000 to -$1000', count: 2876 },
    { range: '-$1000 to $0', count: 5123 },
    { range: '$0 to $1000', count: 5334 },
    { range: '$1000 to $2000', count: 2765 },
    { range: '$2000 to $3000', count: 1231 },
    { range: '$3000 to $4000', count: 765 },
    { range: '$4000 to $5000', count: 398 },
    { range: '>$5000', count: 267 }
  ];

  // Radar chart data for vehicle types
  const vehicleTypeRadarData = [
    { type: 'SUV', depreciation: 18, popularity: 92, premium: 82, maintenance: 65, fuel_efficiency: 50 },
    { type: 'Sedan', depreciation: 25, popularity: 85, premium: 60, maintenance: 75, fuel_efficiency: 85 },
    { type: 'Truck', depreciation: 15, popularity: 78, premium: 75, maintenance: 55, fuel_efficiency: 35 },
    { type: 'Hatchback', depreciation: 22, popularity: 70, premium: 55, maintenance: 80, fuel_efficiency: 88 },
    { type: 'Coupe', depreciation: 35, popularity: 50, premium: 65, maintenance: 60, fuel_efficiency: 72 },
    { type: 'Convertible', depreciation: 40, popularity: 35, premium: 68, maintenance: 50, fuel_efficiency: 65 }
  ];

  // Vehicle age depreciation data
  const ageDepreciationData = [
    { age: 1, retainedValue: 85 },
    { age: 2, retainedValue: 75 },
    { age: 3, retainedValue: 67 },
    { age: 4, retainedValue: 60 },
    { age: 5, retainedValue: 54 },
    { age: 6, retainedValue: 49 },
    { age: 7, retainedValue: 45 },
    { age: 8, retainedValue: 41 },
    { age: 9, retainedValue: 38 },
    { age: 10, retainedValue: 35 },
    { age: 11, retainedValue: 32 },
    { age: 12, retainedValue: 30 },
    { age: 13, retainedValue: 28 },
    { age: 14, retainedValue: 26 },
    { age: 15, retainedValue: 24 }
  ];

  // COLORS
  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8', '#82ca9d'];

  return (
    <div className="p-4 bg-gray-50 min-h-screen">
      <h1 className="text-2xl font-bold text-center mb-6">Used Car Price Analysis: Interactive Visualizations</h1>
      
      {/* Tab Navigation */}
      <div className="flex flex-wrap justify-center mb-6 gap-2">
        {[
          { id: 'featureImportance', label: 'Feature Importance' },
          { id: 'yearPrice', label: 'Year vs Price' },
          { id: 'mileagePrice', label: 'Mileage vs Price' },
          { id: 'manufacturerComparison', label: 'Manufacturer Comparison' },
          { id: 'modelPerformance', label: 'Model Performance' },
          { id: 'conditionEffect', label: 'Condition Effect' },
          { id: 'residuals', label: 'Residuals Distribution' },
          { id: 'vehicleType', label: 'Vehicle Type Analysis' },
          { id: 'depreciation', label: 'Depreciation Curve' }
        ].map(tab => (
          <button
            key={tab.id}
            className={`px-4 py-2 rounded-md transition-all ${
              activeTab === tab.id 
                ? 'bg-blue-600 text-white' 
                : 'bg-white text-gray-700 hover:bg-gray-100'
            }`}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </div>
      
      {/* Visualization Container */}
      <div className="bg-white rounded-lg shadow-md p-4 mb-8">
        <div className="h-96">
          {/* Feature Importance Chart */}
          {activeTab === 'featureImportance' && (
            <div>
              <h2 className="text-xl font-semibold mb-4 text-center">Feature Importance in Predicting Used Car Prices</h2>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  layout="vertical"
                  data={featureImportanceData}
                  margin={{ top: 20, right: 30, left: 100, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" domain={[0, 30]} />
                  <YAxis dataKey="name" type="category" width={100} />
                  <Tooltip formatter={(value) => [`${value.toFixed(1)}%`, 'Importance']} />
                  <Bar dataKey="importance" fill="#8884d8" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
              <p className="text-gray-600 text-sm mt-4 text-center">
                Vehicle age and odometer reading are the most influential factors, accounting for nearly 50% of the price prediction power.
              </p>
            </div>
          )}
          
          {/* Year vs Price Chart */}
          {activeTab === 'yearPrice' && (
            <div>
              <h2 className="text-xl font-semibold mb-4 text-center">Average Price by Vehicle Year</h2>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={yearPriceData}
                  margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="year" />
                  <YAxis domain={[0, 40000]} tickFormatter={(value) => `$${value.toLocaleString()}`} />
                  <Tooltip formatter={(value) => [`$${value.toLocaleString()}`, 'Average Price']} />
                  <Line type="monotone" dataKey="avgPrice" stroke="#8884d8" activeDot={{ r: 8 }} />
                </LineChart>
              </ResponsiveContainer>
              <p className="text-gray-600 text-sm mt-4 text-center">
                Vehicle prices increase steadily with newer model years, showing an average appreciation of 10% per year.
              </p>
            </div>
          )}
          
          {/* Mileage vs Price Chart */}
          {activeTab === 'mileagePrice' && (
            <div>
              <h2 className="text-xl font-semibold mb-4 text-center">Average Price by Mileage Range</h2>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={mileagePriceData}
                  margin={{ top: 20, right: 30, left: 20, bottom: 10 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="mileageBin" angle={-45} textAnchor="end" height={60} />
                  <YAxis tickFormatter={(value) => `$${value.toLocaleString()}`} />
                  <Tooltip formatter={(value) => [`$${value.toLocaleString()}`, 'Average Price']} />
                  <Bar dataKey="avgPrice" fill="#82ca9d" />
                </BarChart>
              </ResponsiveContainer>
              <p className="text-gray-600 text-sm mt-4 text-center">
                A clear negative relationship exists between mileage and price, with vehicles losing approximately 4-7% value per 10,000 miles.
              </p>
            </div>
          )}
          
          {/* Manufacturer Comparison */}
          {activeTab === 'manufacturerComparison' && (
            <div>
              <h2 className="text-xl font-semibold mb-4 text-center">Manufacturer Price Comparison</h2>
              <ResponsiveContainer width="100%" height="100%">
                <ScatterChart
                  margin={{ top: 20, right: 20, bottom: 20, left: 20 }}
                >
                  <CartesianGrid />
                  <XAxis type="number" dataKey="volume" name="Market Share (%)" unit="%" />
                  <YAxis type="number" dataKey="avgPrice" name="Average Price" tickFormatter={(value) => `$${value.toLocaleString()}`} />
                  <ZAxis range={[100, 500]} />
                  <Tooltip cursor={{ strokeDasharray: '3 3' }} formatter={(value, name) => {
                    if (name === 'Market Share (%)') return [`${value}%`, name];
                    return [`$${value.toLocaleString()}`, name];
                  }} />
                  <Legend />
                  {manufacturerPriceData.map((entry, index) => (
                    <Scatter 
                      key={`scatter-${index}`} 
                      name={entry.name} 
                      data={[entry]} 
                      fill={COLORS[index % COLORS.length]} 
                    />
                  ))}
                </ScatterChart>
              </ResponsiveContainer>
              <p className="text-gray-600 text-sm mt-4 text-center">
                Luxury brands (BMW, Mercedes-Benz) command higher prices but have lower market share, while mainstream brands (Toyota, Honda) balance price with popularity.
              </p>
            </div>
          )}
          
          {/* Model Performance */}
          {activeTab === 'modelPerformance' && (
            <div>
              <h2 className="text-xl font-semibold mb-4 text-center">Regression Model Performance Comparison</h2>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={modelPerformanceData}
                  margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="model" />
                  <YAxis yAxisId="left" orientation="left" stroke="#8884d8" />
                  <YAxis yAxisId="right" orientation="right" stroke="#82ca9d" domain={[0, 1]} />
                  <Tooltip />
                  <Legend />
                  <Bar yAxisId="right" dataKey="r2" name="R² Score" fill="#8884d8" />
                  <Bar yAxisId="left" dataKey="rmse" name="RMSE ($)" fill="#82ca9d" />
                </BarChart>
              </ResponsiveContainer>
              <p className="text-gray-600 text-sm mt-4 text-center">
                Tree-based models (XGBoost, Gradient Boosting, Random Forest) outperform linear models in both R² score and error metrics.
              </p>
            </div>
          )}
          
          {/* Condition Effect */}
          {activeTab === 'conditionEffect' && (
            <div>
              <h2 className="text-xl font-semibold mb-4 text-center">Price Effect by Vehicle Condition</h2>
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={conditionPriceData}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    outerRadius={120}
                    fill="#8884d8"
                    dataKey="percentage"
                    nameKey="condition"
                    label={({name, percent}) => `${name}: ${(percent * 100).toFixed(0)}%`}
                  >
                    {conditionPriceData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip formatter={(value, name, props) => {
                    if (props.dataKey === "percentage") {
                      return [`${value}%`, "Market Share"];
                    }
                    return [value, name];
                  }} />
                </PieChart>
              </ResponsiveContainer>
              <div className="grid grid-cols-5 gap-2 mt-2">
                {conditionPriceData.map((entry, index) => (
                  <div key={index} className="text-center">
                    <div className="font-semibold">{entry.condition}</div>
                    <div className="text-sm">
                      Price: {entry.priceMultiplier * 100}%
                    </div>
                  </div>
                ))}
              </div>
              <p className="text-gray-600 text-sm mt-2 text-center">
                Moving from "Good" to "Excellent" condition adds approximately 12% to vehicle value, while "Poor" condition reduces value by 38% compared to "Excellent".
              </p>
            </div>
          )}
          
          {/* Residuals Distribution */}
          {activeTab === 'residuals' && (
            <div>
              <h2 className="text-xl font-semibold mb-4 text-center">Model Residuals Distribution</h2>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={residualDistribution}
                  margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="range" angle={-45} textAnchor="end" height={60} />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="count" fill="#8884d8" />
                </BarChart>
              </ResponsiveContainer>
              <p className="text-gray-600 text-sm mt-4 text-center">
                The residuals are approximately normally distributed around zero, indicating an unbiased model. Most predictions (78%) fall within ±$2,000 of the actual price.
              </p>
            </div>
          )}
          
          {/* Vehicle Type Radar Analysis */}
          {activeTab === 'vehicleType' && (
            <div>
              <h2 className="text-xl font-semibold mb-4 text-center">Vehicle Type Analysis</h2>
              <ResponsiveContainer width="100%" height="100%">
                <RadarChart outerRadius={150} width={730} height={400} data={vehicleTypeRadarData}>
                  <PolarGrid />
                  <PolarAngleAxis dataKey="type" />
                  <PolarRadiusAxis angle={30} domain={[0, 100]} />
                  <Radar name="Depreciation Resistance" dataKey="depreciation" stroke="#8884d8" fill="#8884d8" fillOpacity={0.2} />
                  <Radar name="Market Popularity" dataKey="popularity" stroke="#82ca9d" fill="#82ca9d" fillOpacity={0.2} />
                  <Radar name="Price Premium" dataKey="premium" stroke="#ffc658" fill="#ffc658" fillOpacity={0.2} />
                  <Legend />
                  <Tooltip />
                </RadarChart>
              </ResponsiveContainer>
              <p className="text-gray-600 text-sm mt-4 text-center">
                SUVs and trucks show stronger depreciation resistance and higher price premiums, while sedans and hatchbacks offer better fuel efficiency and lower maintenance costs.
              </p>
            </div>
          )}
          
          {/* Depreciation Curve */}
          {activeTab === 'depreciation' && (
            <div>
              <h2 className="text-xl font-semibold mb-4 text-center">Vehicle Depreciation Curve</h2>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={ageDepreciationData}
                  margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="age" label={{ value: 'Vehicle Age (Years)', position: 'insideBottomRight', offset: -10 }} />
                  <YAxis label={{ value: 'Retained Value (%)', angle: -90, position: 'insideLeft' }} />
                  <Tooltip formatter={(value) => [`${value}%`, 'Retained Value']} />
                  <Line type="monotone" dataKey="retainedValue" stroke="#ff7300" activeDot={{ r: 8 }} />
                </LineChart>
              </ResponsiveContainer>
              <p className="text-gray-600 text-sm mt-4 text-center">
                The steepest depreciation occurs in the first 4 years, with vehicles losing approximately 40% of their value. After 10 years, most vehicles retain only about 35% of their original value.
              </p>
            </div>
          )}
        </div>
      </div>
      
      {/* Key Insights Section */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-xl font-semibold mb-4 text-center">Key Insights from Visualizations</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="border rounded-lg p-4 bg-blue-50">
            <h3 className="text-lg font-medium mb-2">Depreciation Patterns</h3>
            <p className="text-gray-700">
              Vehicles lose approximately 15-25% of their value in the first year, and 50% by year 5. The depreciation curve flattens after 7 years, making 5-7 year old vehicles a potential value sweet spot for buyers.
            </p>
          </div>
          <div className="border rounded-lg p-4 bg-green-50">
            <h3 className="text-lg font-medium mb-2">Mileage Impact</h3>
            <p className="text-gray-700">
              Each 10,000 miles reduces vehicle value by approximately 4-7%, with the effect being more pronounced in luxury vehicles. Low-mileage vehicles (under 30,000 miles) command significant premiums.
            </p>
          </div>
          <div className="border rounded-lg p-4 bg-yellow-50">
            <h3 className="text-lg font-medium mb-2">Brand Value Differences</h3>
            <p className="text-gray-700">
              Luxury brands maintain higher values but depreciate faster in absolute dollars. Japanese brands (Toyota, Honda) demonstrate stronger value retention among non-luxury vehicles.
            </p>
          </div>
          <div className="border rounded-lg p-4 bg-purple-50">
            <h3 className="text-lg font-medium mb-2">Model Performance</h3>
            <p className="text-gray-700">
              Tree-based models outperform linear models by a significant margin, suggesting complex non-linear relationships in vehicle pricing. XGBoost provides the best balance of accuracy and efficiency.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default UsedCarVisualization;