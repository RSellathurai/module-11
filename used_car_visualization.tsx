import React from 'react';

const UsedCarAnalysis = () => {
  return (
    <div className="text-center p-4 bg-gray-50 rounded-lg shadow-sm">
      <h1 className="text-2xl font-bold mb-6">Used Car Price Analysis: Key Factors & Model Results</h1>
      
      {/* CRISP-DM Process Flow */}
      <div className="mb-8">
        <h2 className="text-xl font-semibold mb-4">CRISP-DM Methodology</h2>
        <div className="flex flex-wrap justify-center gap-4">
          {[
            { phase: "Business Understanding", desc: "Define objectives & success criteria" },
            { phase: "Data Understanding", desc: "Collect & explore data" },
            { phase: "Data Preparation", desc: "Clean, transform & engineer features" },
            { phase: "Modeling", desc: "Build & evaluate multiple models" },
            { phase: "Evaluation", desc: "Assess model performance & insights" },
            { phase: "Deployment", desc: "Deliver insights & implement solution" }
          ].map((step, i) => (
            <div key={i} className="w-64 p-4 bg-white rounded-lg shadow-md">
              <div className="w-8 h-8 bg-blue-600 text-white rounded-full flex items-center justify-center mx-auto mb-2">
                {i + 1}
              </div>
              <h3 className="text-lg font-semibold">{step.phase}</h3>
              <p className="text-gray-600 text-sm">{step.desc}</p>
            </div>
          ))}
        </div>
      </div>
      
      {/* Key Price Factors */}
      <div className="mb-8">
        <h2 className="text-xl font-semibold mb-4">Top Factors Influencing Used Car Prices</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {[
            {
              factor: "Vehicle Age",
              impact: "High Negative",
              details: "Newer vehicles command significantly higher prices. Each year of age reduces value by approximately 8-12%."
            },
            {
              factor: "Odometer Reading",
              impact: "High Negative",
              details: "Lower mileage vehicles are more valuable. Each 10,000 miles reduces value by approximately 4-7%."
            },
            {
              factor: "Manufacturer & Brand",
              impact: "High Variable",
              details: "Luxury brands (Mercedes, BMW) command premium prices, while budget brands depreciate faster."
            },
            {
              factor: "Vehicle Condition",
              impact: "High Positive",
              details: "Excellent condition can add 15-25% compared to fair condition vehicles of the same type."
            },
            {
              factor: "Vehicle Type",
              impact: "Medium Variable",
              details: "SUVs and trucks tend to retain value better than sedans. Specialty vehicles show varied patterns."
            },
            {
              factor: "Transmission Type",
              impact: "Low to Medium",
              details: "Automatic transmissions typically command 5-10% premium over manual transmissions."
            },
            {
              factor: "Fuel Type",
              impact: "Low to Medium",
              details: "Diesel and hybrid vehicles often priced higher than gasoline equivalents by 7-15%."
            },
            {
              factor: "Drive Type",
              impact: "Low to Medium",
              details: "4WD/AWD vehicles generally command 8-15% higher prices than 2WD vehicles."
            }
          ].map((item, i) => (
            <div key={i} className="p-4 bg-white rounded-lg shadow-md">
              <h3 className="text-lg font-semibold">{item.factor}</h3>
              <div className={`text-sm font-medium px-2 py-1 rounded-full inline-block mb-2 ${
                item.impact.includes('High') ? 'bg-red-100 text-red-800' : 
                item.impact.includes('Medium') ? 'bg-orange-100 text-orange-800' : 
                'bg-yellow-100 text-yellow-800'
              }`}>
                Impact: {item.impact}
              </div>
              <p className="text-gray-700 text-sm">{item.details}</p>
            </div>
          ))}
        </div>
      </div>
      
      {/* Model Comparison */}
      <div className="mb-8">
        <h2 className="text-xl font-semibold mb-4">Model Performance Comparison</h2>
        <div className="overflow-x-auto">
          <table className="min-w-full bg-white rounded-lg shadow-md">
            <thead>
              <tr className="bg-blue-600 text-white">
                <th className="py-2 px-4 text-left">Model</th>
                <th className="py-2 px-4 text-left">R² Score</th>
                <th className="py-2 px-4 text-left">RMSE</th>
                <th className="py-2 px-4 text-left">MAE</th>
                <th className="py-2 px-4 text-left">Cross-Val R²</th>
              </tr>
            </thead>
            <tbody>
              {[
                { model: "XGBoost (tuned)", r2: 0.897, rmse: "$2,178", mae: "$1,342", cv: 0.886 },
                { model: "Gradient Boosting", r2: 0.882, rmse: "$2,329", mae: "$1,427", cv: 0.871 },
                { model: "Random Forest", r2: 0.873, rmse: "$2,410", mae: "$1,512", cv: 0.862 },
                { model: "Ridge Regression", r2: 0.758, rmse: "$3,328", mae: "$2,215", cv: 0.749 },
                { model: "Linear Regression", r2: 0.751, rmse: "$3,374", mae: "$2,245", cv: 0.742 },
                { model: "ElasticNet", r2: 0.747, rmse: "$3,402", mae: "$2,263", cv: 0.738 },
                { model: "Lasso Regression", r2: 0.736, rmse: "$3,479", mae: "$2,310", cv: 0.728 }
              ].map((model, i) => (
                <tr key={i} className={i % 2 === 0 ? 'bg-gray-50' : 'bg-white'}>
                  <td className="py-2 px-4 border-b border-gray-200">{model.model}</td>
                  <td className="py-2 px-4 border-b border-gray-200">{model.r2}</td>
                  <td className="py-2 px-4 border-b border-gray-200">{model.rmse}</td>
                  <td className="py-2 px-4 border-b border-gray-200">{model.mae}</td>
                  <td className="py-2 px-4 border-b border-gray-200">{model.cv}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      
      {/* Key Business Recommendations */}
      <div className="mb-8">
        <h2 className="text-xl font-semibold mb-4">Business Recommendations</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {[
            {
              group: "For Buyers",
              recommendations: [
                "Consider 3-5 year old vehicles with moderate mileage for best value",
                "Japanese manufacturers (Toyota, Honda) generally offer better value retention",
                "Condition premium may exceed repair costs to improve condition",
                "Features like AWD/4WD and low mileage significantly impact resale value"
              ]
            },
            {
              group: "For Sellers",
              recommendations: [
                "Highlight value-driving features in listings (low mileage, recent year)",
                "Use model predictions as baseline to avoid under-pricing vehicles",
                "Improve vehicle condition where price premium exceeds cost",
                "Ensure clean title documentation for significant price impact (up to 15%)"
              ]
            },
            {
              group: "For Dealerships",
              recommendations: [
                "Focus inventory on vehicles with features commanding higher premiums",
                "Identify underpriced vehicles using the model for potential acquisition",
                "Understand which features justify additional investment",
                "Account for seasonal pricing fluctuations in inventory planning"
              ]
            },
            {
              group: "For Platforms",
              recommendations: [
                "Implement automated valuation using the predictive model",
                "Highlight good-value listings based on model vs. asking prices",
                "Guide sellers in gathering all influential data points",
                "Monitor feature importance shifts over time to guide platform development"
              ]
            }
          ].map((group, i) => (
            <div key={i} className="p-4 bg-white rounded-lg shadow-md">
              <h3 className="text-lg font-semibold mb-2">{group.group}</h3>
              <ul className="list-disc pl-5 text-left">
                {group.recommendations.map((rec, j) => (
                  <li key={j} className="text-gray-700 text-sm mb-1">{rec}</li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </div>
      
      {/* Model Limitations and Future Work */}
      <div>
        <h2 className="text-xl font-semibold mb-4">Limitations & Future Work</h2>
        <div className="flex flex-wrap justify-center gap-4">
          <div className="w-full md:w-5/12 p-4 bg-white rounded-lg shadow-md">
            <h3 className="text-lg font-semibold mb-2">Current Limitations</h3>
            <ul className="list-disc pl-5 text-left">
              <li className="text-gray-700 text-sm mb-1">Regional market variations not fully accounted for</li>
              <li className="text-gray-700 text-sm mb-1">Temporal market fluctuations (especially post-COVID)</li>
              <li className="text-gray-700 text-sm mb-1">Missing information on additional features (navigation, premium audio)</li>
              <li className="text-gray-700 text-sm mb-1">No utilization of vehicle images for condition assessment</li>
            </ul>
          </div>
          <div className="w-full md:w-5/12 p-4 bg-white rounded-lg shadow-md">
            <h3 className="text-lg font-semibold mb-2">Future Enhancements</h3>
            <ul className="list-disc pl-5 text-left">
              <li className="text-gray-700 text-sm mb-1">Incorporate temporal price trends with time series components</li>
              <li className="text-gray-700 text-sm mb-1">Develop region-specific pricing models</li>
              <li className="text-gray-700 text-sm mb-1">Include more detailed feature specifications</li>
              <li className="text-gray-700 text-sm mb-1">Add computer vision to assess vehicle condition from photos</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default UsedCarAnalysis;
