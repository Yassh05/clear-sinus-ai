import { useState, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Loader2, Stethoscope, ChevronDown, ChevronUp, Upload, X, FileText, Image as ImageIcon } from "lucide-react";
import { supabase } from "@/integrations/supabase/client";
import { toast } from "sonner";

const SYMPTOM_CATEGORIES = [
  {
    category: "Primary Symptoms",
    symptoms: [
      { id: "nasal_congestion", label: "Nasal Congestion / Blockage", description: "Difficulty breathing through nose" },
      { id: "facial_pain", label: "Facial Pain / Pressure", description: "Around eyes, cheeks, or forehead" },
      { id: "nasal_discharge", label: "Nasal Discharge", description: "Thick, discolored mucus" },
      { id: "reduced_smell", label: "Reduced Sense of Smell", description: "Partial or complete loss" },
    ],
  },
  {
    category: "Secondary Symptoms",
    symptoms: [
      { id: "headache", label: "Headache", description: "Persistent or recurring" },
      { id: "post_nasal_drip", label: "Post-Nasal Drip", description: "Mucus dripping down throat" },
      { id: "cough", label: "Cough", description: "Especially at night" },
      { id: "ear_pressure", label: "Ear Pressure / Fullness", description: "Blocked or full sensation" },
    ],
  },
  {
    category: "Additional Indicators",
    symptoms: [
      { id: "fever", label: "Fever", description: "Elevated temperature" },
      { id: "fatigue", label: "Fatigue / Malaise", description: "General tiredness" },
      { id: "dental_pain", label: "Upper Dental Pain", description: "Pain in upper teeth/jaw" },
      { id: "bad_breath", label: "Bad Breath (Halitosis)", description: "Persistent bad odor" },
      { id: "snoring", label: "Snoring / Sleep Issues", description: "Difficulty sleeping due to nasal issues" },
      { id: "nosebleeds", label: "Nosebleeds", description: "Recurring epistaxis" },
    ],
  },
];

const DURATION_OPTIONS = [
  { value: "less_than_week", label: "< 1 week" },
  { value: "1_to_4_weeks", label: "1–4 weeks" },
  { value: "1_to_3_months", label: "1–3 months" },
  { value: "more_than_3_months", label: "3+ months" },
];

const SEVERITY_OPTIONS = [
  { value: "mild", label: "Mild", color: "bg-success/20 text-success border-success/30" },
  { value: "moderate", label: "Moderate", color: "bg-warning/20 text-warning border-warning/30" },
  { value: "severe", label: "Severe", color: "bg-destructive/20 text-destructive border-destructive/30" },
];

const GENDER_OPTIONS = [
  { value: "male", label: "Male" },
  { value: "female", label: "Female" },
  { value: "other", label: "Other" },
];

const ENVIRONMENT_OPTIONS = [
  { value: "urban_pollution", label: "Urban / High Pollution" },
  { value: "rural_clean", label: "Rural / Clean Air" },
  { value: "industrial", label: "Industrial Area" },
  { value: "dry_climate", label: "Dry / Arid Climate" },
  { value: "humid_climate", label: "Humid / Tropical" },
];

export interface SymptomFormData {
  symptoms: string[];
  duration: string;
  severity: string;
  age: string;
  gender: string;
  allergies: boolean;
  smoking: boolean;
  previous_sinus_history: boolean;
  environment: string;
  medications: string;
  report_urls: string[];
}

interface SymptomFormProps {
  onSubmit: (data: SymptomFormData) => void;
  isLoading: boolean;
}

export function SymptomForm({ onSubmit, isLoading }: SymptomFormProps) {
  const [selectedSymptoms, setSelectedSymptoms] = useState<string[]>([]);
  const [duration, setDuration] = useState("");
  const [severity, setSeverity] = useState("");
  const [age, setAge] = useState("");
  const [gender, setGender] = useState("");
  const [allergies, setAllergies] = useState(false);
  const [smoking, setSmoking] = useState(false);
  const [previousHistory, setPreviousHistory] = useState(false);
  const [environment, setEnvironment] = useState("");
  const [medications, setMedications] = useState("");
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState<{ name: string; url: string; type: string }[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    const allowed = ["application/pdf", "image/jpeg", "image/png", "image/webp"];
    const maxSize = 10 * 1024 * 1024; // 10MB

    setIsUploading(true);
    try {
      for (const file of Array.from(files)) {
        if (!allowed.includes(file.type)) {
          toast.error(`${file.name}: Only PDF, JPG, PNG, WEBP files are supported.`);
          continue;
        }
        if (file.size > maxSize) {
          toast.error(`${file.name}: File too large (max 10MB).`);
          continue;
        }

        const ext = file.name.split(".").pop();
        const path = `${crypto.randomUUID()}.${ext}`;
        const { error } = await supabase.storage.from("medical-reports").upload(path, file);
        if (error) {
          toast.error(`Failed to upload ${file.name}`);
          continue;
        }

        const { data: urlData } = supabase.storage.from("medical-reports").getPublicUrl(path);
        setUploadedFiles((prev) => [...prev, { name: file.name, url: urlData.publicUrl, type: file.type }]);
      }
      toast.success("Reports uploaded successfully!");
    } catch {
      toast.error("Upload failed. Please try again.");
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  const removeFile = (url: string) => {
    setUploadedFiles((prev) => prev.filter((f) => f.url !== url));
  };

  const toggleSymptom = (id: string) => {
    setSelectedSymptoms((prev) => (prev.includes(id) ? prev.filter((s) => s !== id) : [...prev, id]));
  };

  const canSubmit = selectedSymptoms.length > 0 && duration && severity && age && gender;

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!canSubmit) return;
    onSubmit({
      symptoms: selectedSymptoms,
      duration,
      severity,
      age,
      gender,
      allergies,
      smoking,
      previous_sinus_history: previousHistory,
      environment,
      medications,
      report_urls: uploadedFiles.map((f) => f.url),
    });
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-8">
      {/* Pipeline indicator */}
      <div className="flex items-center gap-2 px-4 py-3 rounded-lg bg-surface-sunken border border-border opacity-0 animate-fade-up">
        <div className="flex items-center gap-1.5">
          {["Input", "Preprocess", "Feature Eng.", "ML Model", "Predict"].map((step, i) => (
            <span key={step} className="flex items-center gap-1.5 text-[10px] font-medium text-muted-foreground">
              <span className="w-5 h-5 rounded-full bg-primary/15 text-primary flex items-center justify-center text-[10px] font-bold">{i + 1}</span>
              {step}
              {i < 4 && <span className="text-border mx-0.5">→</span>}
            </span>
          ))}
        </div>
      </div>

      {/* Symptoms */}
      {SYMPTOM_CATEGORIES.map((cat, catIdx) => (
        <div
          key={cat.category}
          className="opacity-0 animate-fade-up"
          style={{ animationDelay: `${catIdx * 100 + 100}ms` }}
        >
          <h3 className="font-display text-lg font-semibold text-foreground mb-3">{cat.category}</h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {cat.symptoms.map((symptom) => {
              const selected = selectedSymptoms.includes(symptom.id);
              return (
                <button
                  type="button"
                  key={symptom.id}
                  onClick={() => toggleSymptom(symptom.id)}
                  className={`group relative text-left p-4 rounded-lg border-2 transition-all duration-200 active:scale-[0.98] ${
                    selected
                      ? "border-primary bg-primary/5 shadow-md shadow-primary/10"
                      : "border-border bg-card hover:border-primary/40 hover:shadow-sm"
                  }`}
                >
                  <div className="flex items-start gap-3">
                    <div
                      className={`mt-0.5 w-5 h-5 rounded-md border-2 flex items-center justify-center transition-all duration-200 ${
                        selected ? "bg-primary border-primary" : "border-muted-foreground/30"
                      }`}
                    >
                      {selected && (
                        <svg className="w-3 h-3 text-primary-foreground" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3}>
                          <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                        </svg>
                      )}
                    </div>
                    <div>
                      <p className="font-medium text-sm text-foreground">{symptom.label}</p>
                      <p className="text-xs text-muted-foreground mt-0.5">{symptom.description}</p>
                    </div>
                  </div>
                </button>
              );
            })}
          </div>
        </div>
      ))}

      {/* Duration */}
      <div className="opacity-0 animate-fade-up" style={{ animationDelay: "400ms" }}>
        <h3 className="font-display text-lg font-semibold text-foreground mb-3">Duration of Symptoms</h3>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {DURATION_OPTIONS.map((opt) => (
            <button
              type="button"
              key={opt.value}
              onClick={() => setDuration(opt.value)}
              className={`p-3 rounded-lg border-2 text-sm font-medium transition-all duration-200 active:scale-[0.98] ${
                duration === opt.value
                  ? "border-primary bg-primary/5 text-primary"
                  : "border-border bg-card text-foreground hover:border-primary/40"
              }`}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </div>

      {/* Severity */}
      <div className="opacity-0 animate-fade-up" style={{ animationDelay: "500ms" }}>
        <h3 className="font-display text-lg font-semibold text-foreground mb-3">Symptom Severity</h3>
        <div className="grid grid-cols-3 gap-3">
          {SEVERITY_OPTIONS.map((opt) => (
            <button
              type="button"
              key={opt.value}
              onClick={() => setSeverity(opt.value)}
              className={`p-3 rounded-lg border-2 text-sm font-medium transition-all duration-200 active:scale-[0.98] ${
                severity === opt.value ? `${opt.color} border-current` : "border-border bg-card text-foreground hover:border-primary/40"
              }`}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </div>

      {/* Demographics */}
      <div className="opacity-0 animate-fade-up grid grid-cols-1 sm:grid-cols-3 gap-4" style={{ animationDelay: "600ms" }}>
        <div>
          <h3 className="font-display text-sm font-semibold text-foreground mb-2">Age</h3>
          <input
            type="number"
            min="1"
            max="120"
            value={age}
            onChange={(e) => setAge(e.target.value)}
            placeholder="Enter age"
            className="w-full p-3 rounded-lg border-2 border-border bg-card text-foreground placeholder:text-muted-foreground focus:border-primary focus:outline-none transition-colors text-sm"
          />
        </div>
        <div>
          <h3 className="font-display text-sm font-semibold text-foreground mb-2">Gender</h3>
          <div className="grid grid-cols-3 gap-2">
            {GENDER_OPTIONS.map((opt) => (
              <button
                type="button"
                key={opt.value}
                onClick={() => setGender(opt.value)}
                className={`p-3 rounded-lg border-2 text-xs font-medium transition-all active:scale-[0.98] ${
                  gender === opt.value
                    ? "border-primary bg-primary/5 text-primary"
                    : "border-border bg-card text-foreground hover:border-primary/40"
                }`}
              >
                {opt.label}
              </button>
            ))}
          </div>
        </div>
        <div>
          <h3 className="font-display text-sm font-semibold text-foreground mb-2">Allergy History</h3>
          <button
            type="button"
            onClick={() => setAllergies(!allergies)}
            className={`w-full p-3 rounded-lg border-2 text-sm font-medium transition-all active:scale-[0.98] ${
              allergies
                ? "border-primary bg-primary/5 text-primary"
                : "border-border bg-card text-foreground hover:border-primary/40"
            }`}
          >
            {allergies ? "✓ Known allergies" : "No known allergies"}
          </button>
        </div>
      </div>

      {/* Advanced factors toggle */}
      <div className="opacity-0 animate-fade-up" style={{ animationDelay: "700ms" }}>
        <button
          type="button"
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="flex items-center gap-2 text-sm font-medium text-primary hover:text-primary/80 transition-colors"
        >
          {showAdvanced ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          {showAdvanced ? "Hide" : "Show"} Additional Risk Factors
        </button>

        {showAdvanced && (
          <div className="mt-4 space-y-4 p-4 rounded-xl border border-border bg-surface-sunken">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div>
                <h4 className="text-sm font-semibold text-foreground mb-2">Smoking Status</h4>
                <button
                  type="button"
                  onClick={() => setSmoking(!smoking)}
                  className={`w-full p-3 rounded-lg border-2 text-sm font-medium transition-all active:scale-[0.98] ${
                    smoking
                      ? "border-destructive bg-destructive/5 text-destructive"
                      : "border-border bg-card text-foreground hover:border-primary/40"
                  }`}
                >
                  {smoking ? "✓ Smoker / Ex-smoker" : "Non-smoker"}
                </button>
              </div>
              <div>
                <h4 className="text-sm font-semibold text-foreground mb-2">Previous Sinus History</h4>
                <button
                  type="button"
                  onClick={() => setPreviousHistory(!previousHistory)}
                  className={`w-full p-3 rounded-lg border-2 text-sm font-medium transition-all active:scale-[0.98] ${
                    previousHistory
                      ? "border-warning bg-warning/5 text-warning"
                      : "border-border bg-card text-foreground hover:border-primary/40"
                  }`}
                >
                  {previousHistory ? "✓ Has previous history" : "No previous history"}
                </button>
              </div>
            </div>

            <div>
              <h4 className="text-sm font-semibold text-foreground mb-2">Environmental Exposure</h4>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
                {ENVIRONMENT_OPTIONS.map((opt) => (
                  <button
                    type="button"
                    key={opt.value}
                    onClick={() => setEnvironment(environment === opt.value ? "" : opt.value)}
                    className={`p-2.5 rounded-lg border-2 text-xs font-medium transition-all active:scale-[0.98] ${
                      environment === opt.value
                        ? "border-primary bg-primary/5 text-primary"
                        : "border-border bg-card text-foreground hover:border-primary/40"
                    }`}
                  >
                    {opt.label}
                  </button>
                ))}
              </div>
            </div>

            <div>
              <h4 className="text-sm font-semibold text-foreground mb-2">Current Medications (Optional)</h4>
              <input
                type="text"
                value={medications}
                onChange={(e) => setMedications(e.target.value)}
                placeholder="e.g., Antihistamines, Nasal sprays, Antibiotics..."
                className="w-full p-3 rounded-lg border-2 border-border bg-card text-foreground placeholder:text-muted-foreground focus:border-primary focus:outline-none transition-colors text-sm"
              />
            </div>
          </div>
        )}
      </div>

      {/* Submit */}
      <div className="opacity-0 animate-fade-up pt-4" style={{ animationDelay: "800ms" }}>
        <Button
          type="submit"
          disabled={!canSubmit || isLoading}
          size="lg"
          className="w-full h-14 text-base font-semibold rounded-xl shadow-lg shadow-primary/20 hover:shadow-xl hover:shadow-primary/30 transition-all duration-300 active:scale-[0.98]"
        >
          {isLoading ? (
            <>
              <Loader2 className="animate-spin" />
              Running ML Pipeline...
            </>
          ) : (
            <>
              <Stethoscope />
              Run Prediction Models
            </>
          )}
        </Button>
        {!canSubmit && (
          <p className="text-xs text-muted-foreground text-center mt-3">
            Select symptoms, duration, severity, age, and gender to proceed.
          </p>
        )}
      </div>
    </form>
  );
}
