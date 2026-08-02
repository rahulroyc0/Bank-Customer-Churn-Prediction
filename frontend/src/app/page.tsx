'use client';

import { useState } from 'react';

interface CustomerFormData {
  credit_score: number;
  country: string;
  gender: string;
  age: number;
  tenure: number;
  balance: number;
  products_number: number;
  credit_card: number;
  active_member: number;
  estimated_salary: number;
}

interface PredictionResult {
  churn_prediction: number;
  churn_probability: number;
  risk_level: string;
}

export default function ChurnDashboard() {
  const [formData, setFormData] = useState<CustomerFormData>({
    credit_score: 650,
    country: 'Germany',
    gender: 'Male',
    age: 45,
    tenure: 3,
    balance: 75000,
    products_number: 1,
    credit_card: 1,
    active_member: 1,
    estimated_salary: 50000,
  });

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value, type } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: type === 'number' ? parseFloat(value) || 0 : value,
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);   

    try {
      const response = await fetch('https://bank-customer-churn-prediction-ow0a.onrender.com/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        throw new Error(`API Error: ${response.statusText}`);
      }

      const data: PredictionResult = await response.json();
      setResult(data);
    } catch (err: any) {
      setError(err.message || 'Failed to communicate with FastAPI backend.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-slate-900 text-slate-100 p-8">
      <div className="max-w-6xl mx-auto space-y-8">
        
        {/* HEADER */}
        <header className="border-b border-slate-800 pb-4">
          <h1 className="text-3xl font-bold text-blue-400">AI Banking Churn Intelligence</h1>
          <p className="text-slate-400 text-sm mt-1">Real-time XGBoost ML Risk Engine</p>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          
          {/* FORM SECTION */}
          <form onSubmit={handleSubmit} className="lg:col-span-2 bg-slate-800/50 p-6 rounded-xl border border-slate-700/50 space-y-4">
            <h2 className="text-xl font-semibold mb-4 text-slate-200">Customer Details</h2>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-xs text-slate-400 mb-1">Credit Score</label>
                <input
                  type="number"
                  name="credit_score"
                  value={formData.credit_score}
                  onChange={handleChange}
                  className="w-full bg-slate-900 border border-slate-700 rounded p-2 text-sm text-slate-100 focus:outline-none focus:border-blue-500"
                  required
                />
              </div>

              <div>
                <label className="block text-xs text-slate-400 mb-1">Country</label>
                <select
                  name="country"
                  value={formData.country}
                  onChange={handleChange}
                  className="w-full bg-slate-900 border border-slate-700 rounded p-2 text-sm text-slate-100 focus:outline-none focus:border-blue-500"
                >
                  <option value="France">France</option>
                  <option value="Germany">Germany</option>
                  <option value="Spain">Spain</option>
                </select>
              </div>

              <div>
                <label className="block text-xs text-slate-400 mb-1">Gender</label>
                <select
                  name="gender"
                  value={formData.gender}
                  onChange={handleChange}
                  className="w-full bg-slate-900 border border-slate-700 rounded p-2 text-sm text-slate-100 focus:outline-none focus:border-blue-500"
                >
                  <option value="Male">Male</option>
                  <option value="Female">Female</option>
                </select>
              </div>

              <div>
                <label className="block text-xs text-slate-400 mb-1">Age</label>
                <input
                  type="number"
                  name="age"
                  value={formData.age}
                  onChange={handleChange}
                  className="w-full bg-slate-900 border border-slate-700 rounded p-2 text-sm text-slate-100 focus:outline-none focus:border-blue-500"
                  required
                />
              </div>

              <div>
                <label className="block text-xs text-slate-400 mb-1">Tenure (Years)</label>
                <input
                  type="number"
                  name="tenure"
                  value={formData.tenure}
                  onChange={handleChange}
                  className="w-full bg-slate-900 border border-slate-700 rounded p-2 text-sm text-slate-100 focus:outline-none focus:border-blue-500"
                  required
                />
              </div>

              <div>
                <label className="block text-xs text-slate-400 mb-1">Account Balance ($)</label>
                <input
                  type="number"
                  name="balance"
                  value={formData.balance}
                  onChange={handleChange}
                  className="w-full bg-slate-900 border border-slate-700 rounded p-2 text-sm text-slate-100 focus:outline-none focus:border-blue-500"
                  required
                />
              </div>

              <div>
                <label className="block text-xs text-slate-400 mb-1">Number of Products</label>
                <input
                  type="number"
                  name="products_number"
                  value={formData.products_number}
                  onChange={handleChange}
                  className="w-full bg-slate-900 border border-slate-700 rounded p-2 text-sm text-slate-100 focus:outline-none focus:border-blue-500"
                  required
                />
              </div>

              <div>
                <label className="block text-xs text-slate-400 mb-1">Estimated Salary ($)</label>
                <input
                  type="number"
                  name="estimated_salary"
                  value={formData.estimated_salary}
                  onChange={handleChange}
                  className="w-full bg-slate-900 border border-slate-700 rounded p-2 text-sm text-slate-100 focus:outline-none focus:border-blue-500"
                  required
                />
              </div>

              <div>
                <label className="block text-xs text-slate-400 mb-1">Holds Credit Card</label>
                <select
                  name="credit_card"
                  value={formData.credit_card}
                  onChange={handleChange}
                  className="w-full bg-slate-900 border border-slate-700 rounded p-2 text-sm text-slate-100 focus:outline-none focus:border-blue-500"
                >
                  <option value={1}>Yes</option>
                  <option value={0}>No</option>
                </select>
              </div>

              <div>
                <label className="block text-xs text-slate-400 mb-1">Active Member</label>
                <select
                  name="active_member"
                  value={formData.active_member}
                  onChange={handleChange}
                  className="w-full bg-slate-900 border border-slate-700 rounded p-2 text-sm text-slate-100 focus:outline-none focus:border-blue-500"
                >
                  <option value={1}>Yes</option>
                  <option value={0}>No</option>
                </select>
              </div>
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full mt-6 bg-blue-600 hover:bg-blue-500 text-white font-medium py-2.5 rounded transition duration-200 disabled:opacity-50"
            >
              {loading ? 'Evaluating Risk...' : 'Run Churn Prediction'}
            </button>
          </form>

          {/* PREDICTION DISPLAY SECTION */}
          <div className="bg-slate-800/50 p-6 rounded-xl border border-slate-700/50 flex flex-col justify-between">
            <div>
              <h2 className="text-xl font-semibold mb-4 text-slate-200">Risk Analysis</h2>

              {error && (
                <div className="p-4 bg-red-900/30 border border-red-500/50 rounded-lg text-red-300 text-sm">
                  {error}
                </div>
              )}

              {!result && !error && (
                <div className="text-center text-slate-500 my-16 text-sm">
                  Fill in customer attributes and click <br />
                  <strong className="text-slate-400">Run Churn Prediction</strong>
                </div>
              )}

              {result && (
                <div className="space-y-6">
                  {/* RISK BADGE */}
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-slate-400">Risk Assessment</span>
                    <span
                      className={`px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider ${
                        result.churn_prediction === 1
                          ? 'bg-red-500/20 text-red-400 border border-red-500/30'
                          : 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                      }`}
                    >
                      {result.risk_level}
                    </span>
                  </div>

                  {/* PROBABILITY METER */}
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span className="text-slate-400">Churn Probability</span>
                      <span className="font-semibold text-slate-200">
                        {(result.churn_probability * 100).toFixed(2)}%
                      </span>
                    </div>
                    <div className="w-full bg-slate-900 rounded-full h-3 overflow-hidden border border-slate-700">
                      <div
                        className={`h-full transition-all duration-500 ${
                          result.churn_prediction === 1 ? 'bg-red-500' : 'bg-emerald-500'
                        }`}
                        style={{ width: `${result.churn_probability * 100}%` }}
                      ></div>
                    </div>
                  </div>

                  {/* DETAILS CARD */}
                  <div className="p-4 bg-slate-900/60 rounded-lg border border-slate-700/50 space-y-2 text-xs">
                    <div className="flex justify-between">
                      <span className="text-slate-400">Prediction Output:</span>
                      <span className="text-slate-200 font-mono">
                        {result.churn_prediction === 1 ? '1 (Churn Expected)' : '0 (Likely to Stay)'}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Model Engine:</span>
                      <span className="text-slate-200 font-mono">Tuned XGBoost</span>
                    </div>
                  </div>
                </div>
              )}
            </div>

            <div className="text-xs text-slate-500 text-center border-t border-slate-800 pt-4 mt-6">
              Powered by FastAPI & Next.js App Router
            </div>
          </div>

        </div>
      </div>
    </main>
  );
}