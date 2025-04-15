import React from "react";
import { Button } from "./components/ui/button";
import { Input } from "./components/ui/input";
import { Label } from "./components/ui/label";
import { Card, CardContent } from "./components/ui/card";
import { useState } from "react";
import { motion } from "framer-motion";
// import { Venus, Mars } from "lucide-react";
import { UserIcon, UserCircleIcon } from "lucide-react";
import { useRef } from "react";


export default function BreastCancerDiagnosis() {
  const [step, setStep] = useState(1);
  const [gender, setGender] = useState(null);
  const [patientDetails, setPatientDetails] = useState({
    name: "",
    age: "",
    contact: "",
    weight: "",
    height: "",
    familyHistory: "",
    symptoms: "",
    medicalConditions: ""
  });
  const [image, setImage] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const fileInputRef = useRef(null);
  const [selectedFile, setSelectedFile] = useState(null);

  const handleGenderSelect = (selectedGender) => {
    setGender(selectedGender);
    setStep(2);
  };

  const handlePatientDetailChange = (e) => {
    setPatientDetails({
      ...patientDetails,
      [e.target.name]: e.target.value
    });
  };

  const proceedToUpload = () => {
    const { name, age, contact, weight, height } = patientDetails;
  
    if (!name || !age || !contact || !weight || !height) {
      alert("Please fill in all required fields: Name, Age, Contact, Weight, and Height");
      return;
    }
  
    setStep(3);
  };
  
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
    setLoading(false);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handlePredict = () => {
    if (!selectedFile) return;
  
    setLoading(true);
    const formData = new FormData();
    formData.append("image", selectedFile);
  
    fetch("http://localhost:5000/predict", {
      method: "POST",
      body: formData,
    })
      .then((res) => res.json())
      .then((data) => {
        setResult(data.prediction);
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
            <CardContent className="p-6 flex flex-col items-center gap-6">
              <p className="text-pink-700 font-semibold">Select Your Gender</p>
              <div className="flex justify-center gap-10">
                <motion.div
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.95 }}
                  className="cursor-pointer"
                  onClick={() => handleGenderSelect("female")}
                >
                  <div className="flex flex-col items-center gap-2">
                    {/* <Venus size={48} className="text-pink-600" /> */}
                    <UserIcon size={48} className="text-pink-600" />
                    <span className="text-pink-600 font-medium">Female</span>
                  </div>
                </motion.div>
                <motion.div
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.95 }}
                  className="cursor-pointer"
                  onClick={() => handleGenderSelect("male")}
                >
                  <div className="flex flex-col items-center gap-2">
                    {/* <Mars size={48} className="text-pink-600" /> */}
                    <UserCircleIcon size={48} className="text-pink-600" />
                    <span className="text-pink-600 font-medium">Male</span>
                  </div>
                </motion.div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {step === 2 && (
        <div className="max-w-xl mx-auto">
          <Card className="rounded-2xl shadow-md bg-white">
            <CardContent className="p-6 flex flex-col gap-4">
              <p className="text-pink-700 font-semibold text-lg text-center mb-4">Patient Details</p>
              <div className="flex flex-col gap-4">
                <div>
                  <Label htmlFor="name" className="text-pink-600">Name *</Label>
                  <Input name="name" value={patientDetails.name} onChange={handlePatientDetailChange} placeholder="Enter your name" />
                </div>
                <div>
                  <Label htmlFor="age" className="text-pink-600">Age *</Label>
                  <Input name="age" type="number" value={patientDetails.age} onChange={handlePatientDetailChange} placeholder="Enter your age" />
                </div>
                <div>
                  <Label htmlFor="contact" className="text-pink-600">Contact *</Label>
                  <Input name="contact" value={patientDetails.contact} onChange={handlePatientDetailChange} placeholder="Enter your contact number" />
                </div>
                <div>
                  <Label htmlFor="weight" className="text-pink-600">Weight (kg) *</Label>
                  <Input name="weight" value={patientDetails.weight} onChange={handlePatientDetailChange} placeholder="Enter your weight" />
                </div>
                <div>
                  <Label htmlFor="height" className="text-pink-600">Height (cm) *</Label>
                  <Input name="height" value={patientDetails.height} onChange={handlePatientDetailChange} placeholder="Enter your height" />
                </div>
                <div>
                  <Label htmlFor="familyHistory" className="text-pink-600">Family History</Label>
                  <Input name="familyHistory" value={patientDetails.familyHistory} onChange={handlePatientDetailChange} placeholder="e.g., Mother diagnosed at 45" />
                </div>
                <div>
                  <Label htmlFor="symptoms" className="text-pink-600">Current Symptoms</Label>
                  <Input name="symptoms" value={patientDetails.symptoms} onChange={handlePatientDetailChange} placeholder="e.g., Lump, pain, discharge" />
                </div>
                <div>
                  <Label htmlFor="medicalConditions" className="text-pink-600">Past Medical Conditions</Label>
                  <Input name="medicalConditions" value={patientDetails.medicalConditions} onChange={handlePatientDetailChange} placeholder="e.g., Diabetes, Hypertension" />
                </div>
                <Button onClick={proceedToUpload} className="mt-4 bg-pink-600 hover:bg-pink-700 text-white rounded-lg px-6 py-2 shadow-sm">
                  Continue to Upload
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {step === 3 && (
        <div className="max-w-xl mx-auto">
          <Card className="rounded-2xl shadow-md bg-white">
            <CardContent className="p-6 flex flex-col gap-4">
              <label className="text-pink-700 font-semibold">Upload Mammogram Image</label>
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
                </motion.div>
              )}
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
