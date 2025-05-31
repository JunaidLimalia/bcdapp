import React from "react";
import { Button } from "./components/ui/button";
import { Input } from "./components/ui/input";
import { Label } from "./components/ui/label";
import { Card, CardContent } from "./components/ui/card";
import { useState } from "react";
import { motion } from "framer-motion";
import { UserIcon, UserCircleIcon } from "lucide-react";
import { useRef } from "react";


export default function BreastCancerDiagnosis() {
  const [step, setStep] = useState(1);
  const [image, setImage] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const fileInputRef = useRef(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [gradcam, setGradcam] = useState(null);
  const [superimposed, setSuperimposed] = useState(null);
  const [textExplanation, setTextExplanation] = useState("");
  const [ablationcam, setAblationcam] = useState(null);
  const [ablationSuperimposed, setAblationSuperimposed] = useState(null);
  
  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;
  
    setImage(URL.createObjectURL(file));
    setSelectedFile(file);
    setResult(null);
  };

  const handleClear = () => {
    setImage(null);
    setResult(null);
    setGradcam(null);
    setSuperimposed(null);
    setAblationcam(null);
    setAblationSuperimposed(null);
    setLoading(false);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleNewDiagnosis = () => {
    setImage(null);
    setResult(null);
    setGradcam(null);
    setSuperimposed(null);
    setAblationcam(null);
    setAblationSuperimposed(null);
    setLoading(false);
    setStep(3);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handlePredict = () => {
    if (!selectedFile) return;
  
    setLoading(true);
    const formData = new FormData();
    formData.append("image", selectedFile);
    
    // fetch("http://127.0.0.1:8080/predict", { // to run locally, uncomment this line, comment the next line, run nmp build again
    fetch("https://bcdapp-358860318763.asia-southeast1.run.app/predict", {
      method: "POST",
      body: formData,
    })
      .then((res) => res.json())
      .then((data) => {
        setResult(data.prediction);
        setGradcam(data.gradcam);
        setSuperimposed(data.superimposed);
        setAblationcam(data.ablationcam);
        setAblationSuperimposed(data.ablationSuperimposed);
        setTextExplanation(data.textExplanation);
        setLoading(false);
      })
      .catch((err) => {
        console.error("Prediction failed:", err);
        setResult("Error during prediction");
        setLoading(false);
      });
  };
  
  return (
    <div className="min-h-screen bg-pink-50 p-8">
      <motion.h1
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-4xl font-bold text-center text-pink-700 mb-8"
      >
        Breast Cancer Diagnosis
      </motion.h1>
      {step === 1 && (
        <div className="max-w-xl mx-auto">
          <Card className="rounded-2xl shadow-md bg-white">
            <CardContent className="p-6 flex justify-center">
              <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                <Button
                  onClick={() => setStep(3)}
                  className="bg-pink-600 hover:bg-pink-700 text-white rounded-lg px-6 py-2"
                >
                  Start Diagnosis
                </Button>
              </motion.div>
            </CardContent>
          </Card>
        </div>
      )}

      {step === 3 && (
        <div className="max-w-xl mx-auto">
          <Card className="rounded-2xl shadow-md bg-white">
            <CardContent className="p-6 flex flex-col gap-4">
              <label className="text-pink-700 font-semibold">Upload Microscopic Image of Breast Tumor</label>
              <Input type="file" accept="image/*" onChange={handleImageUpload} ref={fileInputRef} className="rounded-lg border border-pink-300 p-2 text-sm file:mr-4 file:rounded-lg file:border-0 file:bg-pink-600 file:py-2 file:px-4 file:text-white file:cursor-pointer hover:file:bg-pink-700"/>
              {image && (
                <div className="border-2 border-pink-200 rounded-xl mt-4 p-4 flex flex-col items-center gap-4">
                  <img
                    src={image}
                    alt="Uploaded Preview"
                    className="rounded-xl border border-pink-300 max-h-96 object-contain"
                  />
                  <div className="flex gap-4">
                    <Button onClick={handlePredict} className="bg-pink-600 hover:bg-pink-700 text-white rounded-lg px-4 py-2 shadow-sm">
                      {loading ? "Predicting..." : "Predict"}
                    </Button>
                    <Button onClick={handleClear} variant="outline" className="border border-pink-400 text-pink-600 rounded-lg px-4 py-2">
                      Clear
                    </Button>
                  </div>
                </div>
              )}
              {loading && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="text-center text-pink-500 font-medium mt-4"
                >
                  Processing image, please wait...
                </motion.div>
              )}
              {result && !loading && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="mt-4 text-center text-lg font-bold text-pink-600"
                >
                  Diagnosis Result: {result}

                  {(result === "Benign" || result === "Malignant") && (
                    <div className="mt-4">
                      <Button
                        onClick={() => setStep(4)}
                        className="bg-pink-600 hover:bg-pink-700 text-white rounded-lg px-6 py-2 mt-2"
                      >
                        View Explanation
                      </Button>
                    </div>
                  )}

                  {result === "Irrelevant" && (
                    <div className="mt-4 text-pink-600 text-base font-medium">
                      The uploaded image is not relevant for diagnosis. <br />
                      Please clear the image and upload a valid <strong>microscopic image of a breast tumor</strong> for accurate analysis.
                    </div>
                  )}
                </motion.div>
              )}
            </CardContent>
          </Card>
        </div>
      )}

      {step === 4 && (
      <div className="max-w-4xl mx-auto">
        <Card className="rounded-2xl shadow-md bg-white">
          <CardContent className="p-8">
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="text-center mb-6"
            >
              <h2 className="text-2xl font-bold text-pink-700 mb-2">
                Diagnosis Result: {result}
              </h2>
              <p className="text-pink-600 font-semibold text-md whitespace-pre-line mb-4">
                {textExplanation}
              </p>
            </motion.div>

            <div className="bg-pink-100 rounded-xl p-6 mb-8 text-sm text-gray-800 max-w-3xl mx-auto shadow-sm leading-relaxed">
              <h3 className="text-lg font-semibold text-pink-700 mb-4">Understanding the Visualizations</h3>
              <p className="mb-3">
                The images shown below (called <strong>heatmaps</strong>) reveal what parts of your mammogram the AI model focused on to make its decision.
              </p>
              <ul className="list-disc pl-5 space-y-2 mb-4">
                <li>
                  <strong>Red/Yellow areas</strong> represent regions that strongly influenced the model's diagnosis.
                </li>
                <li>
                  <strong>Blue areas</strong> had less impact on the decision-making process.
                </li>
              </ul>
              <p className="mb-2">
                Two types of visualizations are provided:
              </p>
              <ul className="list-disc pl-5 space-y-2">
                <li>
                  <strong>Grad-CAM</strong>: Highlights which image regions the model considers important based on internal attention.
                </li>
                <li>
                  <strong>Ablation-CAM</strong>: Shows regions that cause a noticeable change in prediction when altered.
                </li>
              </ul>
            </div>

            <div className="text-center mb-4">
              <h3 className="text-pink-600 font-semibold text-lg">Grad-CAM Visualization</h3>
            </div>

            <div className="flex justify-center gap-6 flex-wrap mb-8">
              <img
                src={gradcam}
                alt="Grad-CAM Heatmap"
                className="rounded-xl border border-pink-300 w-[350px] h-[350px] object-cover"
              />
              <img
                src={superimposed}
                alt="Superimposed Grad-CAM"
                className="rounded-xl border border-pink-300 w-[350px] h-[350px] object-cover"
              />
            </div>

            {ablationcam && ablationSuperimposed && (
              <>
                <div className="text-center mb-4">
                  <h3 className="text-pink-600 font-semibold text-lg">Ablation-CAM Visualization</h3>
                </div>

                <div className="flex justify-center gap-6 flex-wrap mb-8">
                  <img
                    src={ablationcam}
                    alt="Ablation-CAM Heatmap"
                    className="rounded-xl border border-pink-300 w-[350px] h-[350px] object-cover"
                  />
                  <img
                    src={ablationSuperimposed}
                    alt="Superimposed Ablation-CAM"
                    className="rounded-xl border border-pink-300 w-[350px] h-[350px] object-cover"
                  />
                </div>
              </>
            )}

            <div className="flex justify-center mt-4">
              <Button
                onClick={handleNewDiagnosis}
                className="bg-pink-600 hover:bg-pink-700 text-white rounded-lg px-6 py-2"
              >
                New Diagnosis
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    )}
    </div>
  );
}