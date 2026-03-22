import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Loader2, Stethoscope } from "lucide-react";

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
  { value: "less_than_week", label: "Less than 1 week" },
  { value: "1_to_4_weeks", label: "1–4 weeks" },
  { value: "1_to_3_months", label: "1–3 months" },
  { value: "more_than_3_months", label: "More than 3 months" },
];

const SEVERITY_OPTIONS = [
  { value: "mild", label: "Mild", color: "bg-success/20 text-success border-success/30" },
  { value: "moderate", label: "Moderate", color: "bg-warning/20 text-warning border-warning/30" },
  { value: "severe", label: "Severe", color: "bg-destructive/20 text-destructive border-destructive/30" },
];

interface SymptomFormProps {
  onSubmit: (data: { symptoms: string[]; duration: string; severity: string; age: string; allergies: boolean }) => void;
  isLoading: boolean;
}

export function SymptomForm({ onSubmit, isLoading }: SymptomFormProps) {
  const [selectedSymptoms, setSelectedSymptoms] = useState<string[]>([]);
  const [duration, setDuration] = useState("");
  const [severity, setSeverity] = useState("");
  const [age, setAge] = useState("");
  const [allergies, setAllergies] = useState(false);

  const toggleSymptom = (id: string) => {
    setSelectedSymptoms((prev) => (prev.includes(id) ? prev.filter((s) => s !== id) : [...prev, id]));
  };

  const canSubmit = selectedSymptoms.length > 0 && duration && severity && age;

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!canSubmit) return;
    onSubmit({ symptoms: selectedSymptoms, duration, severity, age, allergies });
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-8">
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

      {/* Age + Allergies */}
      <div className="opacity-0 animate-fade-up grid grid-cols-1 sm:grid-cols-2 gap-6" style={{ animationDelay: "600ms" }}>
        <div>
          <h3 className="font-display text-lg font-semibold text-foreground mb-3">Age</h3>
          <input
            type="number"
            min="1"
            max="120"
            value={age}
            onChange={(e) => setAge(e.target.value)}
            placeholder="Enter your age"
            className="w-full p-3 rounded-lg border-2 border-border bg-card text-foreground placeholder:text-muted-foreground focus:border-primary focus:outline-none transition-colors"
          />
        </div>
        <div>
          <h3 className="font-display text-lg font-semibold text-foreground mb-3">Allergy History</h3>
          <button
            type="button"
            onClick={() => setAllergies(!allergies)}
            className={`w-full p-3 rounded-lg border-2 text-sm font-medium transition-all duration-200 active:scale-[0.98] ${
              allergies
                ? "border-primary bg-primary/5 text-primary"
                : "border-border bg-card text-foreground hover:border-primary/40"
            }`}
          >
            {allergies ? "Yes, I have known allergies" : "No known allergies"}
          </button>
        </div>
      </div>

      {/* Submit */}
      <div className="opacity-0 animate-fade-up pt-4" style={{ animationDelay: "700ms" }}>
        <Button
          type="submit"
          disabled={!canSubmit || isLoading}
          size="lg"
          className="w-full h-14 text-base font-semibold rounded-xl shadow-lg shadow-primary/20 hover:shadow-xl hover:shadow-primary/30 transition-all duration-300 active:scale-[0.98]"
        >
          {isLoading ? (
            <>
              <Loader2 className="animate-spin" />
              Analyzing Symptoms...
            </>
          ) : (
            <>
              <Stethoscope />
              Analyze & Predict
            </>
          )}
        </Button>
        {!canSubmit && (
          <p className="text-xs text-muted-foreground text-center mt-3">
            Please select at least one symptom, duration, severity, and enter your age.
          </p>
        )}
      </div>
    </form>
  );
}
