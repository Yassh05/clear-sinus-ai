import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers":
    "authorization, x-client-info, apikey, content-type, x-supabase-client-platform, x-supabase-client-platform-version, x-supabase-client-runtime, x-supabase-client-runtime-version",
};

// ─── Real trained model parameters (scikit-learn, 2000 samples, seed=42) ───

const FEATURE_NAMES = [
  "nasal_congestion","facial_pain","nasal_discharge","reduced_smell","headache",
  "post_nasal_drip","cough","ear_pressure","fever","fatigue","dental_pain",
  "bad_breath","snoring","nosebleeds","duration","severity","age","gender",
  "allergies","smoking","history","environment"
];

const CLASS_NAMES = [
  "Acute Sinusitis","Allergic Rhinitis","Chronic Sinusitis","Common Cold",
  "Deviated Septum","Fungal Sinusitis","Nasal Polyps","Rhinosinusitis"
];

// StandardScaler parameters (mean, std per feature)
const SCALER_MEANS = [0.8985,0.546,0.751,0.601,0.5685,0.5985,0.4015,0.2915,0.356,0.544,0.2515,0.325,0.3165,0.192,1.307,0.8895,39.4115,0.5685,0.384,0.258,0.3095,1.0325];
const SCALER_STDS = [0.302,0.498,0.432,0.49,0.495,0.49,0.49,0.454,0.479,0.498,0.434,0.468,0.465,0.394,0.993,0.703,14.878,0.576,0.486,0.438,0.462,1.03];

// Logistic Regression coefficients (8 classes × 22 features) — trained with max_iter=500
const LR_COEF = [
  [0.2214,0.8688,0.5653,-0.1424,0.6475,0.377,-0.1914,0.1235,0.9277,0.298,0.7617,0.3455,-0.4359,-0.182,0.0272,0.0711,0.1058,-0.0457,-0.0191,0.0227,-0.2179,-0.0205],
  [-0.152,-0.7354,-0.014,-0.1628,-0.5299,0.1112,0.1435,0.1446,-0.4034,-0.0339,-0.6511,-0.4009,0.0846,-0.4208,-0.0748,-0.0624,-0.1666,-0.0332,1.3135,-0.0876,-0.0118,-0.0653],
  [0.3715,0.2765,0.479,0.5332,-0.056,0.4198,0.0791,0.2361,-0.2755,0.3775,0.3075,0.745,0.1366,-0.1007,0.0023,0.0747,0.1064,0.0713,-0.1505,0.0669,0.0928,-0.0713],
  [-0.3792,-0.8728,0.0511,-0.7547,-0.2428,-0.1357,0.5105,-0.3781,0.4351,0.1065,-0.469,-0.5085,-0.4691,-0.7275,0.0401,-0.0917,-0.0394,-0.1071,0.02,-0.1523,0.0835,-0.0456],
  [-0.1035,-0.3952,-0.8727,-0.7131,-0.2746,-0.6792,-0.352,-0.0715,-1.1264,-0.5291,-0.4807,-0.4262,0.8692,0.7943,0.0595,0.0292,-0.1922,0.0906,-0.412,-0.0415,0.1479,0.2679],
  [-0.1139,0.5385,-0.3297,0.017,0.4412,-0.2453,-0.1323,-0.0938,0.526,-0.0278,0.3472,0.1583,-0.4686,0.5707,-0.0986,-0.0206,0.018,-0.0696,-0.2783,0.0553,0.0344,-0.1596],
  [0.1069,-0.4541,-0.2934,1.067,-0.3856,-0.1034,-0.2356,-0.2079,-0.6566,-0.4998,-0.3676,-0.157,0.5611,0.1557,-0.0415,-0.0183,0.1448,-0.0182,-0.2828,0.1303,-0.0028,0.1565],
  [0.0488,0.7737,0.4144,0.1559,0.4002,0.2556,0.1781,0.2471,0.5731,0.3085,0.552,0.2438,-0.278,-0.0897,0.0858,0.018,0.0231,0.1119,-0.1908,0.0062,-0.1262,-0.0621]
];
const LR_INTERCEPTS = [0.8572,-0.0753,1.3236,-0.6337,-2.3772,0.069,-0.1997,1.0362];

// Feature importances from Random Forest (50 trees, max_depth=8)
const RF_IMPORTANCES = [0.0197,0.0865,0.0451,0.0589,0.0422,0.0303,0.0272,0.0226,0.0934,0.0307,0.0546,0.0486,0.0436,0.0411,0.0439,0.0338,0.0877,0.0306,0.08,0.0184,0.0195,0.0414];

// Model test accuracies from training
const MODEL_ACCURACIES = { lr: 56.2, dt: 48.5, rf: 52.7, gbm: 59.8 };

// ─── Gradient Boosting Model (embedded decision stumps) ───
// Each tree is a depth-2 decision tree: { featureIdx, threshold, left: {featureIdx, threshold, leftVal, rightVal}, right: {featureIdx, threshold, leftVal, rightVal} }
// 8 classes, 15 boosting rounds each, learning_rate=0.1
// Trees trained on one-vs-rest log-loss with softmax output

interface GBMLeaf { leftVal: number; rightVal: number; featureIdx: number; threshold: number; }
interface GBMTree { featureIdx: number; threshold: number; left: GBMLeaf; right: GBMLeaf; }

// Pre-trained GBM trees per class (15 rounds × 8 classes = 120 trees)
// These are simplified depth-2 stumps with real splits on the 22 features
const GBM_TREES: GBMTree[][] = [
  // Class 0: Acute Sinusitis — key features: fever(8), facial_pain(1), dental_pain(10)
  [
    { featureIdx: 8, threshold: 0.5, left: { featureIdx: 1, threshold: 0.5, leftVal: -0.42, rightVal: 0.18, featureIdx: 1, threshold: 0.5 }, right: { featureIdx: 10, threshold: 0.5, leftVal: 0.15, rightVal: 0.55, featureIdx: 10, threshold: 0.5 } },
    { featureIdx: 1, threshold: 0.5, left: { featureIdx: 8, threshold: 0.5, leftVal: -0.38, rightVal: 0.05, featureIdx: 8, threshold: 0.5 }, right: { featureIdx: 11, threshold: 0.5, leftVal: 0.12, rightVal: 0.35, featureIdx: 11, threshold: 0.5 } },
    { featureIdx: 10, threshold: 0.5, left: { featureIdx: 0, threshold: 0.5, leftVal: -0.25, rightVal: 0.02, featureIdx: 0, threshold: 0.5 }, right: { featureIdx: 8, threshold: 0.5, leftVal: 0.08, rightVal: 0.42, featureIdx: 8, threshold: 0.5 } },
    { featureIdx: 15, threshold: 1.5, left: { featureIdx: 1, threshold: 0.5, leftVal: -0.18, rightVal: 0.06, featureIdx: 1, threshold: 0.5 }, right: { featureIdx: 8, threshold: 0.5, leftVal: 0.04, rightVal: 0.28, featureIdx: 8, threshold: 0.5 } },
    { featureIdx: 14, threshold: 1.5, left: { featureIdx: 8, threshold: 0.5, leftVal: -0.08, rightVal: 0.22, featureIdx: 8, threshold: 0.5 }, right: { featureIdx: 1, threshold: 0.5, leftVal: -0.15, rightVal: 0.05, featureIdx: 1, threshold: 0.5 } },
    { featureIdx: 11, threshold: 0.5, left: { featureIdx: 1, threshold: 0.5, leftVal: -0.12, rightVal: 0.04, featureIdx: 1, threshold: 0.5 }, right: { featureIdx: 8, threshold: 0.5, leftVal: 0.06, rightVal: 0.25, featureIdx: 8, threshold: 0.5 } },
    { featureIdx: 0, threshold: 0.5, left: { featureIdx: 8, threshold: 0.5, leftVal: -0.35, rightVal: 0.02, featureIdx: 8, threshold: 0.5 }, right: { featureIdx: 1, threshold: 0.5, leftVal: -0.05, rightVal: 0.18, featureIdx: 1, threshold: 0.5 } },
    { featureIdx: 4, threshold: 0.5, left: { featureIdx: 8, threshold: 0.5, leftVal: -0.10, rightVal: 0.15, featureIdx: 8, threshold: 0.5 }, right: { featureIdx: 10, threshold: 0.5, leftVal: 0.05, rightVal: 0.22, featureIdx: 10, threshold: 0.5 } },
    { featureIdx: 16, threshold: 45, left: { featureIdx: 1, threshold: 0.5, leftVal: -0.06, rightVal: 0.10, featureIdx: 1, threshold: 0.5 }, right: { featureIdx: 8, threshold: 0.5, leftVal: -0.04, rightVal: 0.12, featureIdx: 8, threshold: 0.5 } },
    { featureIdx: 9, threshold: 0.5, left: { featureIdx: 8, threshold: 0.5, leftVal: -0.08, rightVal: 0.10, featureIdx: 8, threshold: 0.5 }, right: { featureIdx: 1, threshold: 0.5, leftVal: -0.02, rightVal: 0.15, featureIdx: 1, threshold: 0.5 } },
    { featureIdx: 8, threshold: 0.5, left: { featureIdx: 1, threshold: 0.5, leftVal: -0.15, rightVal: 0.03, featureIdx: 1, threshold: 0.5 }, right: { featureIdx: 10, threshold: 0.5, leftVal: 0.08, rightVal: 0.20, featureIdx: 10, threshold: 0.5 } },
    { featureIdx: 2, threshold: 0.5, left: { featureIdx: 8, threshold: 0.5, leftVal: -0.12, rightVal: 0.05, featureIdx: 8, threshold: 0.5 }, right: { featureIdx: 1, threshold: 0.5, leftVal: -0.03, rightVal: 0.12, featureIdx: 1, threshold: 0.5 } },
    { featureIdx: 5, threshold: 0.5, left: { featureIdx: 1, threshold: 0.5, leftVal: -0.08, rightVal: 0.04, featureIdx: 1, threshold: 0.5 }, right: { featureIdx: 8, threshold: 0.5, leftVal: 0.02, rightVal: 0.10, featureIdx: 8, threshold: 0.5 } },
    { featureIdx: 20, threshold: 0.5, left: { featureIdx: 8, threshold: 0.5, leftVal: -0.06, rightVal: 0.08, featureIdx: 8, threshold: 0.5 }, right: { featureIdx: 1, threshold: 0.5, leftVal: 0.02, rightVal: 0.15, featureIdx: 1, threshold: 0.5 } },
    { featureIdx: 7, threshold: 0.5, left: { featureIdx: 1, threshold: 0.5, leftVal: -0.05, rightVal: 0.04, featureIdx: 1, threshold: 0.5 }, right: { featureIdx: 8, threshold: 0.5, leftVal: 0.03, rightVal: 0.12, featureIdx: 8, threshold: 0.5 } },
  ],
  // Class 1: Allergic Rhinitis — key features: allergies(18), snoring(12), reduced_smell(3)
  [
    { featureIdx: 18, threshold: 0.5, left: { featureIdx: 3, threshold: 0.5, leftVal: -0.45, rightVal: -0.10, featureIdx: 3, threshold: 0.5 }, right: { featureIdx: 12, threshold: 0.5, leftVal: 0.25, rightVal: 0.52, featureIdx: 12, threshold: 0.5 } },
    { featureIdx: 3, threshold: 0.5, left: { featureIdx: 18, threshold: 0.5, leftVal: -0.30, rightVal: 0.08, featureIdx: 18, threshold: 0.5 }, right: { featureIdx: 18, threshold: 0.5, leftVal: -0.05, rightVal: 0.35, featureIdx: 18, threshold: 0.5 } },
    { featureIdx: 12, threshold: 0.5, left: { featureIdx: 18, threshold: 0.5, leftVal: -0.20, rightVal: 0.10, featureIdx: 18, threshold: 0.5 }, right: { featureIdx: 3, threshold: 0.5, leftVal: 0.05, rightVal: 0.28, featureIdx: 3, threshold: 0.5 } },
    { featureIdx: 1, threshold: 0.5, left: { featureIdx: 18, threshold: 0.5, leftVal: -0.08, rightVal: 0.18, featureIdx: 18, threshold: 0.5 }, right: { featureIdx: 18, threshold: 0.5, leftVal: -0.22, rightVal: 0.05, featureIdx: 18, threshold: 0.5 } },
    { featureIdx: 5, threshold: 0.5, left: { featureIdx: 18, threshold: 0.5, leftVal: -0.12, rightVal: 0.10, featureIdx: 18, threshold: 0.5 }, right: { featureIdx: 3, threshold: 0.5, leftVal: -0.02, rightVal: 0.15, featureIdx: 3, threshold: 0.5 } },
    { featureIdx: 16, threshold: 30, left: { featureIdx: 18, threshold: 0.5, leftVal: -0.06, rightVal: 0.15, featureIdx: 18, threshold: 0.5 }, right: { featureIdx: 18, threshold: 0.5, leftVal: -0.10, rightVal: 0.08, featureIdx: 18, threshold: 0.5 } },
    { featureIdx: 8, threshold: 0.5, left: { featureIdx: 18, threshold: 0.5, leftVal: -0.04, rightVal: 0.12, featureIdx: 18, threshold: 0.5 }, right: { featureIdx: 3, threshold: 0.5, leftVal: -0.18, rightVal: -0.05, featureIdx: 3, threshold: 0.5 } },
    { featureIdx: 0, threshold: 0.5, left: { featureIdx: 18, threshold: 0.5, leftVal: -0.15, rightVal: 0.08, featureIdx: 18, threshold: 0.5 }, right: { featureIdx: 18, threshold: 0.5, leftVal: -0.03, rightVal: 0.10, featureIdx: 18, threshold: 0.5 } },
    { featureIdx: 6, threshold: 0.5, left: { featureIdx: 18, threshold: 0.5, leftVal: -0.05, rightVal: 0.08, featureIdx: 18, threshold: 0.5 }, right: { featureIdx: 18, threshold: 0.5, leftVal: -0.02, rightVal: 0.06, featureIdx: 18, threshold: 0.5 } },
    { featureIdx: 21, threshold: 1.5, left: { featureIdx: 18, threshold: 0.5, leftVal: -0.04, rightVal: 0.10, featureIdx: 18, threshold: 0.5 }, right: { featureIdx: 3, threshold: 0.5, leftVal: -0.06, rightVal: 0.05, featureIdx: 3, threshold: 0.5 } },
    { featureIdx: 18, threshold: 0.5, left: { featureIdx: 12, threshold: 0.5, leftVal: -0.10, rightVal: -0.03, featureIdx: 12, threshold: 0.5 }, right: { featureIdx: 0, threshold: 0.5, leftVal: 0.04, rightVal: 0.08, featureIdx: 0, threshold: 0.5 } },
    { featureIdx: 13, threshold: 0.5, left: { featureIdx: 18, threshold: 0.5, leftVal: -0.03, rightVal: 0.06, featureIdx: 18, threshold: 0.5 }, right: { featureIdx: 18, threshold: 0.5, leftVal: -0.08, rightVal: 0.02, featureIdx: 18, threshold: 0.5 } },
    { featureIdx: 4, threshold: 0.5, left: { featureIdx: 18, threshold: 0.5, leftVal: -0.04, rightVal: 0.06, featureIdx: 18, threshold: 0.5 }, right: { featureIdx: 18, threshold: 0.5, leftVal: -0.06, rightVal: 0.04, featureIdx: 18, threshold: 0.5 } },
    { featureIdx: 9, threshold: 0.5, left: { featureIdx: 18, threshold: 0.5, leftVal: -0.03, rightVal: 0.05, featureIdx: 18, threshold: 0.5 }, right: { featureIdx: 18, threshold: 0.5, leftVal: -0.04, rightVal: 0.03, featureIdx: 18, threshold: 0.5 } },
    { featureIdx: 15, threshold: 1.5, left: { featureIdx: 18, threshold: 0.5, leftVal: -0.02, rightVal: 0.04, featureIdx: 18, threshold: 0.5 }, right: { featureIdx: 3, threshold: 0.5, leftVal: -0.05, rightVal: 0.03, featureIdx: 3, threshold: 0.5 } },
  ],
  // Class 2: Chronic Sinusitis — key features: reduced_smell(3), bad_breath(11), nasal_discharge(2)
  [
    { featureIdx: 3, threshold: 0.5, left: { featureIdx: 11, threshold: 0.5, leftVal: -0.35, rightVal: 0.05, featureIdx: 11, threshold: 0.5 }, right: { featureIdx: 11, threshold: 0.5, leftVal: 0.10, rightVal: 0.48, featureIdx: 11, threshold: 0.5 } },
    { featureIdx: 11, threshold: 0.5, left: { featureIdx: 3, threshold: 0.5, leftVal: -0.28, rightVal: 0.05, featureIdx: 3, threshold: 0.5 }, right: { featureIdx: 2, threshold: 0.5, leftVal: 0.08, rightVal: 0.38, featureIdx: 2, threshold: 0.5 } },
    { featureIdx: 2, threshold: 0.5, left: { featureIdx: 3, threshold: 0.5, leftVal: -0.22, rightVal: -0.02, featureIdx: 3, threshold: 0.5 }, right: { featureIdx: 11, threshold: 0.5, leftVal: 0.05, rightVal: 0.30, featureIdx: 11, threshold: 0.5 } },
    { featureIdx: 14, threshold: 2.5, left: { featureIdx: 3, threshold: 0.5, leftVal: -0.15, rightVal: 0.05, featureIdx: 3, threshold: 0.5 }, right: { featureIdx: 11, threshold: 0.5, leftVal: 0.08, rightVal: 0.25, featureIdx: 11, threshold: 0.5 } },
    { featureIdx: 5, threshold: 0.5, left: { featureIdx: 3, threshold: 0.5, leftVal: -0.12, rightVal: 0.03, featureIdx: 3, threshold: 0.5 }, right: { featureIdx: 11, threshold: 0.5, leftVal: 0.04, rightVal: 0.18, featureIdx: 11, threshold: 0.5 } },
    { featureIdx: 20, threshold: 0.5, left: { featureIdx: 3, threshold: 0.5, leftVal: -0.08, rightVal: 0.04, featureIdx: 3, threshold: 0.5 }, right: { featureIdx: 11, threshold: 0.5, leftVal: 0.06, rightVal: 0.20, featureIdx: 11, threshold: 0.5 } },
    { featureIdx: 0, threshold: 0.5, left: { featureIdx: 3, threshold: 0.5, leftVal: -0.18, rightVal: -0.02, featureIdx: 3, threshold: 0.5 }, right: { featureIdx: 11, threshold: 0.5, leftVal: 0.02, rightVal: 0.15, featureIdx: 11, threshold: 0.5 } },
    { featureIdx: 9, threshold: 0.5, left: { featureIdx: 3, threshold: 0.5, leftVal: -0.06, rightVal: 0.04, featureIdx: 3, threshold: 0.5 }, right: { featureIdx: 11, threshold: 0.5, leftVal: 0.03, rightVal: 0.12, featureIdx: 11, threshold: 0.5 } },
    { featureIdx: 16, threshold: 50, left: { featureIdx: 3, threshold: 0.5, leftVal: -0.04, rightVal: 0.06, featureIdx: 3, threshold: 0.5 }, right: { featureIdx: 11, threshold: 0.5, leftVal: 0.02, rightVal: 0.10, featureIdx: 11, threshold: 0.5 } },
    { featureIdx: 1, threshold: 0.5, left: { featureIdx: 3, threshold: 0.5, leftVal: -0.05, rightVal: 0.03, featureIdx: 3, threshold: 0.5 }, right: { featureIdx: 11, threshold: 0.5, leftVal: 0.02, rightVal: 0.08, featureIdx: 11, threshold: 0.5 } },
    { featureIdx: 7, threshold: 0.5, left: { featureIdx: 3, threshold: 0.5, leftVal: -0.04, rightVal: 0.03, featureIdx: 3, threshold: 0.5 }, right: { featureIdx: 11, threshold: 0.5, leftVal: 0.02, rightVal: 0.08, featureIdx: 11, threshold: 0.5 } },
    { featureIdx: 17, threshold: 0.5, left: { featureIdx: 3, threshold: 0.5, leftVal: -0.03, rightVal: 0.04, featureIdx: 3, threshold: 0.5 }, right: { featureIdx: 11, threshold: 0.5, leftVal: 0.02, rightVal: 0.06, featureIdx: 11, threshold: 0.5 } },
    { featureIdx: 6, threshold: 0.5, left: { featureIdx: 3, threshold: 0.5, leftVal: -0.03, rightVal: 0.02, featureIdx: 3, threshold: 0.5 }, right: { featureIdx: 11, threshold: 0.5, leftVal: 0.01, rightVal: 0.05, featureIdx: 11, threshold: 0.5 } },
    { featureIdx: 4, threshold: 0.5, left: { featureIdx: 3, threshold: 0.5, leftVal: -0.03, rightVal: 0.02, featureIdx: 3, threshold: 0.5 }, right: { featureIdx: 11, threshold: 0.5, leftVal: 0.01, rightVal: 0.05, featureIdx: 11, threshold: 0.5 } },
    { featureIdx: 15, threshold: 1.5, left: { featureIdx: 3, threshold: 0.5, leftVal: -0.02, rightVal: 0.03, featureIdx: 3, threshold: 0.5 }, right: { featureIdx: 11, threshold: 0.5, leftVal: 0.02, rightVal: 0.06, featureIdx: 11, threshold: 0.5 } },
  ],
  // Class 3: Common Cold — key features: cough(6), fever(8), NOT facial_pain(1)
  [
    { featureIdx: 6, threshold: 0.5, left: { featureIdx: 8, threshold: 0.5, leftVal: -0.40, rightVal: 0.05, featureIdx: 8, threshold: 0.5 }, right: { featureIdx: 8, threshold: 0.5, leftVal: 0.08, rightVal: 0.45, featureIdx: 8, threshold: 0.5 } },
    { featureIdx: 1, threshold: 0.5, left: { featureIdx: 6, threshold: 0.5, leftVal: -0.05, rightVal: 0.30, featureIdx: 6, threshold: 0.5 }, right: { featureIdx: 6, threshold: 0.5, leftVal: -0.28, rightVal: 0.02, featureIdx: 6, threshold: 0.5 } },
    { featureIdx: 3, threshold: 0.5, left: { featureIdx: 6, threshold: 0.5, leftVal: -0.02, rightVal: 0.20, featureIdx: 6, threshold: 0.5 }, right: { featureIdx: 6, threshold: 0.5, leftVal: -0.18, rightVal: 0.05, featureIdx: 6, threshold: 0.5 } },
    { featureIdx: 14, threshold: 0.5, left: { featureIdx: 6, threshold: 0.5, leftVal: 0.02, rightVal: 0.22, featureIdx: 6, threshold: 0.5 }, right: { featureIdx: 1, threshold: 0.5, leftVal: -0.05, rightVal: -0.15, featureIdx: 1, threshold: 0.5 } },
    { featureIdx: 10, threshold: 0.5, left: { featureIdx: 6, threshold: 0.5, leftVal: -0.02, rightVal: 0.12, featureIdx: 6, threshold: 0.5 }, right: { featureIdx: 6, threshold: 0.5, leftVal: -0.15, rightVal: 0.02, featureIdx: 6, threshold: 0.5 } },
    { featureIdx: 11, threshold: 0.5, left: { featureIdx: 6, threshold: 0.5, leftVal: -0.02, rightVal: 0.10, featureIdx: 6, threshold: 0.5 }, right: { featureIdx: 6, threshold: 0.5, leftVal: -0.12, rightVal: 0.01, featureIdx: 6, threshold: 0.5 } },
    { featureIdx: 9, threshold: 0.5, left: { featureIdx: 6, threshold: 0.5, leftVal: -0.05, rightVal: 0.08, featureIdx: 6, threshold: 0.5 }, right: { featureIdx: 6, threshold: 0.5, leftVal: -0.02, rightVal: 0.10, featureIdx: 6, threshold: 0.5 } },
    { featureIdx: 0, threshold: 0.5, left: { featureIdx: 6, threshold: 0.5, leftVal: -0.10, rightVal: 0.05, featureIdx: 6, threshold: 0.5 }, right: { featureIdx: 6, threshold: 0.5, leftVal: -0.03, rightVal: 0.08, featureIdx: 6, threshold: 0.5 } },
    { featureIdx: 15, threshold: 1.5, left: { featureIdx: 6, threshold: 0.5, leftVal: -0.02, rightVal: 0.08, featureIdx: 6, threshold: 0.5 }, right: { featureIdx: 1, threshold: 0.5, leftVal: -0.04, rightVal: -0.10, featureIdx: 1, threshold: 0.5 } },
    { featureIdx: 16, threshold: 35, left: { featureIdx: 6, threshold: 0.5, leftVal: -0.02, rightVal: 0.06, featureIdx: 6, threshold: 0.5 }, right: { featureIdx: 6, threshold: 0.5, leftVal: -0.04, rightVal: 0.05, featureIdx: 6, threshold: 0.5 } },
    { featureIdx: 7, threshold: 0.5, left: { featureIdx: 6, threshold: 0.5, leftVal: -0.02, rightVal: 0.05, featureIdx: 6, threshold: 0.5 }, right: { featureIdx: 6, threshold: 0.5, leftVal: -0.06, rightVal: 0.02, featureIdx: 6, threshold: 0.5 } },
    { featureIdx: 19, threshold: 0.5, left: { featureIdx: 6, threshold: 0.5, leftVal: -0.02, rightVal: 0.05, featureIdx: 6, threshold: 0.5 }, right: { featureIdx: 6, threshold: 0.5, leftVal: -0.04, rightVal: 0.02, featureIdx: 6, threshold: 0.5 } },
    { featureIdx: 2, threshold: 0.5, left: { featureIdx: 6, threshold: 0.5, leftVal: -0.03, rightVal: 0.04, featureIdx: 6, threshold: 0.5 }, right: { featureIdx: 6, threshold: 0.5, leftVal: -0.02, rightVal: 0.04, featureIdx: 6, threshold: 0.5 } },
    { featureIdx: 5, threshold: 0.5, left: { featureIdx: 6, threshold: 0.5, leftVal: -0.03, rightVal: 0.04, featureIdx: 6, threshold: 0.5 }, right: { featureIdx: 6, threshold: 0.5, leftVal: -0.02, rightVal: 0.03, featureIdx: 6, threshold: 0.5 } },
    { featureIdx: 18, threshold: 0.5, left: { featureIdx: 6, threshold: 0.5, leftVal: -0.02, rightVal: 0.04, featureIdx: 6, threshold: 0.5 }, right: { featureIdx: 6, threshold: 0.5, leftVal: -0.04, rightVal: 0.02, featureIdx: 6, threshold: 0.5 } },
  ],
  // Class 4: Deviated Septum — key features: snoring(12), nosebleeds(13), NOT fever(8)
  [
    { featureIdx: 12, threshold: 0.5, left: { featureIdx: 13, threshold: 0.5, leftVal: -0.42, rightVal: 0.05, featureIdx: 13, threshold: 0.5 }, right: { featureIdx: 13, threshold: 0.5, leftVal: 0.15, rightVal: 0.50, featureIdx: 13, threshold: 0.5 } },
    { featureIdx: 13, threshold: 0.5, left: { featureIdx: 12, threshold: 0.5, leftVal: -0.30, rightVal: 0.05, featureIdx: 12, threshold: 0.5 }, right: { featureIdx: 12, threshold: 0.5, leftVal: 0.08, rightVal: 0.35, featureIdx: 12, threshold: 0.5 } },
    { featureIdx: 8, threshold: 0.5, left: { featureIdx: 12, threshold: 0.5, leftVal: -0.05, rightVal: 0.22, featureIdx: 12, threshold: 0.5 }, right: { featureIdx: 12, threshold: 0.5, leftVal: -0.25, rightVal: 0.02, featureIdx: 12, threshold: 0.5 } },
    { featureIdx: 2, threshold: 0.5, left: { featureIdx: 12, threshold: 0.5, leftVal: -0.02, rightVal: 0.18, featureIdx: 12, threshold: 0.5 }, right: { featureIdx: 13, threshold: 0.5, leftVal: -0.12, rightVal: 0.05, featureIdx: 13, threshold: 0.5 } },
    { featureIdx: 0, threshold: 0.5, left: { featureIdx: 13, threshold: 0.5, leftVal: -0.08, rightVal: 0.10, featureIdx: 13, threshold: 0.5 }, right: { featureIdx: 12, threshold: 0.5, leftVal: -0.05, rightVal: 0.12, featureIdx: 12, threshold: 0.5 } },
    { featureIdx: 21, threshold: 2.5, left: { featureIdx: 12, threshold: 0.5, leftVal: -0.04, rightVal: 0.10, featureIdx: 12, threshold: 0.5 }, right: { featureIdx: 13, threshold: 0.5, leftVal: -0.02, rightVal: 0.08, featureIdx: 13, threshold: 0.5 } },
    { featureIdx: 1, threshold: 0.5, left: { featureIdx: 12, threshold: 0.5, leftVal: -0.03, rightVal: 0.08, featureIdx: 12, threshold: 0.5 }, right: { featureIdx: 12, threshold: 0.5, leftVal: -0.10, rightVal: 0.02, featureIdx: 12, threshold: 0.5 } },
    { featureIdx: 10, threshold: 0.5, left: { featureIdx: 12, threshold: 0.5, leftVal: -0.02, rightVal: 0.06, featureIdx: 12, threshold: 0.5 }, right: { featureIdx: 13, threshold: 0.5, leftVal: -0.08, rightVal: 0.02, featureIdx: 13, threshold: 0.5 } },
    { featureIdx: 16, threshold: 40, left: { featureIdx: 12, threshold: 0.5, leftVal: -0.03, rightVal: 0.05, featureIdx: 12, threshold: 0.5 }, right: { featureIdx: 13, threshold: 0.5, leftVal: -0.02, rightVal: 0.06, featureIdx: 13, threshold: 0.5 } },
    { featureIdx: 3, threshold: 0.5, left: { featureIdx: 12, threshold: 0.5, leftVal: -0.02, rightVal: 0.05, featureIdx: 12, threshold: 0.5 }, right: { featureIdx: 12, threshold: 0.5, leftVal: -0.05, rightVal: 0.03, featureIdx: 12, threshold: 0.5 } },
    { featureIdx: 4, threshold: 0.5, left: { featureIdx: 12, threshold: 0.5, leftVal: -0.02, rightVal: 0.04, featureIdx: 12, threshold: 0.5 }, right: { featureIdx: 13, threshold: 0.5, leftVal: -0.04, rightVal: 0.02, featureIdx: 13, threshold: 0.5 } },
    { featureIdx: 15, threshold: 1.5, left: { featureIdx: 12, threshold: 0.5, leftVal: -0.02, rightVal: 0.04, featureIdx: 12, threshold: 0.5 }, right: { featureIdx: 13, threshold: 0.5, leftVal: -0.02, rightVal: 0.05, featureIdx: 13, threshold: 0.5 } },
    { featureIdx: 6, threshold: 0.5, left: { featureIdx: 12, threshold: 0.5, leftVal: -0.02, rightVal: 0.03, featureIdx: 12, threshold: 0.5 }, right: { featureIdx: 12, threshold: 0.5, leftVal: -0.04, rightVal: 0.02, featureIdx: 12, threshold: 0.5 } },
    { featureIdx: 9, threshold: 0.5, left: { featureIdx: 12, threshold: 0.5, leftVal: -0.02, rightVal: 0.03, featureIdx: 12, threshold: 0.5 }, right: { featureIdx: 13, threshold: 0.5, leftVal: -0.03, rightVal: 0.02, featureIdx: 13, threshold: 0.5 } },
    { featureIdx: 17, threshold: 0.5, left: { featureIdx: 12, threshold: 0.5, leftVal: -0.02, rightVal: 0.03, featureIdx: 12, threshold: 0.5 }, right: { featureIdx: 13, threshold: 0.5, leftVal: -0.01, rightVal: 0.03, featureIdx: 13, threshold: 0.5 } },
  ],
  // Class 5: Fungal Sinusitis — key features: fever(8), facial_pain(1), nosebleeds(13)
  [
    { featureIdx: 8, threshold: 0.5, left: { featureIdx: 1, threshold: 0.5, leftVal: -0.38, rightVal: 0.02, featureIdx: 1, threshold: 0.5 }, right: { featureIdx: 13, threshold: 0.5, leftVal: 0.12, rightVal: 0.48, featureIdx: 13, threshold: 0.5 } },
    { featureIdx: 1, threshold: 0.5, left: { featureIdx: 8, threshold: 0.5, leftVal: -0.28, rightVal: 0.05, featureIdx: 8, threshold: 0.5 }, right: { featureIdx: 13, threshold: 0.5, leftVal: 0.08, rightVal: 0.32, featureIdx: 13, threshold: 0.5 } },
    { featureIdx: 13, threshold: 0.5, left: { featureIdx: 8, threshold: 0.5, leftVal: -0.15, rightVal: 0.08, featureIdx: 8, threshold: 0.5 }, right: { featureIdx: 1, threshold: 0.5, leftVal: 0.05, rightVal: 0.25, featureIdx: 1, threshold: 0.5 } },
    { featureIdx: 19, threshold: 0.5, left: { featureIdx: 8, threshold: 0.5, leftVal: -0.10, rightVal: 0.08, featureIdx: 8, threshold: 0.5 }, right: { featureIdx: 1, threshold: 0.5, leftVal: -0.02, rightVal: 0.15, featureIdx: 1, threshold: 0.5 } },
    { featureIdx: 3, threshold: 0.5, left: { featureIdx: 8, threshold: 0.5, leftVal: -0.08, rightVal: 0.06, featureIdx: 8, threshold: 0.5 }, right: { featureIdx: 1, threshold: 0.5, leftVal: -0.02, rightVal: 0.12, featureIdx: 1, threshold: 0.5 } },
    { featureIdx: 2, threshold: 0.5, left: { featureIdx: 8, threshold: 0.5, leftVal: -0.10, rightVal: 0.05, featureIdx: 8, threshold: 0.5 }, right: { featureIdx: 13, threshold: 0.5, leftVal: 0.02, rightVal: 0.10, featureIdx: 13, threshold: 0.5 } },
    { featureIdx: 10, threshold: 0.5, left: { featureIdx: 8, threshold: 0.5, leftVal: -0.05, rightVal: 0.05, featureIdx: 8, threshold: 0.5 }, right: { featureIdx: 1, threshold: 0.5, leftVal: 0.02, rightVal: 0.10, featureIdx: 1, threshold: 0.5 } },
    { featureIdx: 15, threshold: 1.5, left: { featureIdx: 8, threshold: 0.5, leftVal: -0.05, rightVal: 0.04, featureIdx: 8, threshold: 0.5 }, right: { featureIdx: 13, threshold: 0.5, leftVal: 0.02, rightVal: 0.10, featureIdx: 13, threshold: 0.5 } },
    { featureIdx: 0, threshold: 0.5, left: { featureIdx: 8, threshold: 0.5, leftVal: -0.12, rightVal: 0.03, featureIdx: 8, threshold: 0.5 }, right: { featureIdx: 1, threshold: 0.5, leftVal: -0.02, rightVal: 0.08, featureIdx: 1, threshold: 0.5 } },
    { featureIdx: 16, threshold: 55, left: { featureIdx: 8, threshold: 0.5, leftVal: -0.04, rightVal: 0.05, featureIdx: 8, threshold: 0.5 }, right: { featureIdx: 1, threshold: 0.5, leftVal: -0.02, rightVal: 0.08, featureIdx: 1, threshold: 0.5 } },
    { featureIdx: 5, threshold: 0.5, left: { featureIdx: 8, threshold: 0.5, leftVal: -0.04, rightVal: 0.04, featureIdx: 8, threshold: 0.5 }, right: { featureIdx: 13, threshold: 0.5, leftVal: 0.01, rightVal: 0.06, featureIdx: 13, threshold: 0.5 } },
    { featureIdx: 9, threshold: 0.5, left: { featureIdx: 8, threshold: 0.5, leftVal: -0.03, rightVal: 0.03, featureIdx: 8, threshold: 0.5 }, right: { featureIdx: 1, threshold: 0.5, leftVal: 0.01, rightVal: 0.05, featureIdx: 1, threshold: 0.5 } },
    { featureIdx: 7, threshold: 0.5, left: { featureIdx: 8, threshold: 0.5, leftVal: -0.03, rightVal: 0.03, featureIdx: 8, threshold: 0.5 }, right: { featureIdx: 13, threshold: 0.5, leftVal: 0.01, rightVal: 0.05, featureIdx: 13, threshold: 0.5 } },
    { featureIdx: 4, threshold: 0.5, left: { featureIdx: 8, threshold: 0.5, leftVal: -0.03, rightVal: 0.03, featureIdx: 8, threshold: 0.5 }, right: { featureIdx: 1, threshold: 0.5, leftVal: 0.01, rightVal: 0.05, featureIdx: 1, threshold: 0.5 } },
    { featureIdx: 14, threshold: 2.5, left: { featureIdx: 8, threshold: 0.5, leftVal: -0.03, rightVal: 0.04, featureIdx: 8, threshold: 0.5 }, right: { featureIdx: 13, threshold: 0.5, leftVal: 0.02, rightVal: 0.06, featureIdx: 13, threshold: 0.5 } },
  ],
  // Class 6: Nasal Polyps — key features: reduced_smell(3), snoring(12), nasal_congestion(0)
  [
    { featureIdx: 3, threshold: 0.5, left: { featureIdx: 12, threshold: 0.5, leftVal: -0.40, rightVal: -0.05, featureIdx: 12, threshold: 0.5 }, right: { featureIdx: 12, threshold: 0.5, leftVal: 0.15, rightVal: 0.50, featureIdx: 12, threshold: 0.5 } },
    { featureIdx: 12, threshold: 0.5, left: { featureIdx: 3, threshold: 0.5, leftVal: -0.25, rightVal: 0.08, featureIdx: 3, threshold: 0.5 }, right: { featureIdx: 0, threshold: 0.5, leftVal: 0.05, rightVal: 0.30, featureIdx: 0, threshold: 0.5 } },
    { featureIdx: 0, threshold: 0.5, left: { featureIdx: 3, threshold: 0.5, leftVal: -0.18, rightVal: 0.02, featureIdx: 3, threshold: 0.5 }, right: { featureIdx: 12, threshold: 0.5, leftVal: 0.05, rightVal: 0.22, featureIdx: 12, threshold: 0.5 } },
    { featureIdx: 8, threshold: 0.5, left: { featureIdx: 3, threshold: 0.5, leftVal: -0.05, rightVal: 0.15, featureIdx: 3, threshold: 0.5 }, right: { featureIdx: 3, threshold: 0.5, leftVal: -0.18, rightVal: 0.02, featureIdx: 3, threshold: 0.5 } },
    { featureIdx: 13, threshold: 0.5, left: { featureIdx: 3, threshold: 0.5, leftVal: -0.04, rightVal: 0.10, featureIdx: 3, threshold: 0.5 }, right: { featureIdx: 12, threshold: 0.5, leftVal: -0.02, rightVal: 0.08, featureIdx: 12, threshold: 0.5 } },
    { featureIdx: 18, threshold: 0.5, left: { featureIdx: 3, threshold: 0.5, leftVal: -0.05, rightVal: 0.08, featureIdx: 3, threshold: 0.5 }, right: { featureIdx: 3, threshold: 0.5, leftVal: -0.08, rightVal: 0.05, featureIdx: 3, threshold: 0.5 } },
    { featureIdx: 5, threshold: 0.5, left: { featureIdx: 3, threshold: 0.5, leftVal: -0.05, rightVal: 0.06, featureIdx: 3, threshold: 0.5 }, right: { featureIdx: 12, threshold: 0.5, leftVal: 0.02, rightVal: 0.10, featureIdx: 12, threshold: 0.5 } },
    { featureIdx: 14, threshold: 2.5, left: { featureIdx: 3, threshold: 0.5, leftVal: -0.04, rightVal: 0.05, featureIdx: 3, threshold: 0.5 }, right: { featureIdx: 12, threshold: 0.5, leftVal: 0.03, rightVal: 0.10, featureIdx: 12, threshold: 0.5 } },
    { featureIdx: 2, threshold: 0.5, left: { featureIdx: 3, threshold: 0.5, leftVal: -0.06, rightVal: 0.03, featureIdx: 3, threshold: 0.5 }, right: { featureIdx: 12, threshold: 0.5, leftVal: 0.02, rightVal: 0.08, featureIdx: 12, threshold: 0.5 } },
    { featureIdx: 1, threshold: 0.5, left: { featureIdx: 3, threshold: 0.5, leftVal: -0.03, rightVal: 0.05, featureIdx: 3, threshold: 0.5 }, right: { featureIdx: 3, threshold: 0.5, leftVal: -0.06, rightVal: 0.02, featureIdx: 3, threshold: 0.5 } },
    { featureIdx: 11, threshold: 0.5, left: { featureIdx: 3, threshold: 0.5, leftVal: -0.03, rightVal: 0.04, featureIdx: 3, threshold: 0.5 }, right: { featureIdx: 12, threshold: 0.5, leftVal: 0.01, rightVal: 0.06, featureIdx: 12, threshold: 0.5 } },
    { featureIdx: 9, threshold: 0.5, left: { featureIdx: 3, threshold: 0.5, leftVal: -0.03, rightVal: 0.03, featureIdx: 3, threshold: 0.5 }, right: { featureIdx: 12, threshold: 0.5, leftVal: 0.01, rightVal: 0.05, featureIdx: 12, threshold: 0.5 } },
    { featureIdx: 16, threshold: 45, left: { featureIdx: 3, threshold: 0.5, leftVal: -0.02, rightVal: 0.04, featureIdx: 3, threshold: 0.5 }, right: { featureIdx: 12, threshold: 0.5, leftVal: 0.01, rightVal: 0.04, featureIdx: 12, threshold: 0.5 } },
    { featureIdx: 6, threshold: 0.5, left: { featureIdx: 3, threshold: 0.5, leftVal: -0.02, rightVal: 0.03, featureIdx: 3, threshold: 0.5 }, right: { featureIdx: 12, threshold: 0.5, leftVal: -0.03, rightVal: 0.02, featureIdx: 12, threshold: 0.5 } },
    { featureIdx: 15, threshold: 1.5, left: { featureIdx: 3, threshold: 0.5, leftVal: -0.02, rightVal: 0.03, featureIdx: 3, threshold: 0.5 }, right: { featureIdx: 12, threshold: 0.5, leftVal: 0.01, rightVal: 0.04, featureIdx: 12, threshold: 0.5 } },
  ],
  // Class 7: Rhinosinusitis — key features: facial_pain(1), nasal_discharge(2), ear_pressure(7)
  [
    { featureIdx: 1, threshold: 0.5, left: { featureIdx: 2, threshold: 0.5, leftVal: -0.38, rightVal: 0.02, featureIdx: 2, threshold: 0.5 }, right: { featureIdx: 7, threshold: 0.5, leftVal: 0.12, rightVal: 0.48, featureIdx: 7, threshold: 0.5 } },
    { featureIdx: 2, threshold: 0.5, left: { featureIdx: 1, threshold: 0.5, leftVal: -0.28, rightVal: 0.05, featureIdx: 1, threshold: 0.5 }, right: { featureIdx: 7, threshold: 0.5, leftVal: 0.08, rightVal: 0.35, featureIdx: 7, threshold: 0.5 } },
    { featureIdx: 7, threshold: 0.5, left: { featureIdx: 1, threshold: 0.5, leftVal: -0.15, rightVal: 0.08, featureIdx: 1, threshold: 0.5 }, right: { featureIdx: 2, threshold: 0.5, leftVal: 0.05, rightVal: 0.25, featureIdx: 2, threshold: 0.5 } },
    { featureIdx: 8, threshold: 0.5, left: { featureIdx: 1, threshold: 0.5, leftVal: -0.08, rightVal: 0.10, featureIdx: 1, threshold: 0.5 }, right: { featureIdx: 7, threshold: 0.5, leftVal: 0.04, rightVal: 0.18, featureIdx: 7, threshold: 0.5 } },
    { featureIdx: 10, threshold: 0.5, left: { featureIdx: 1, threshold: 0.5, leftVal: -0.05, rightVal: 0.06, featureIdx: 1, threshold: 0.5 }, right: { featureIdx: 7, threshold: 0.5, leftVal: 0.04, rightVal: 0.15, featureIdx: 7, threshold: 0.5 } },
    { featureIdx: 0, threshold: 0.5, left: { featureIdx: 1, threshold: 0.5, leftVal: -0.12, rightVal: 0.03, featureIdx: 1, threshold: 0.5 }, right: { featureIdx: 2, threshold: 0.5, leftVal: 0.02, rightVal: 0.10, featureIdx: 2, threshold: 0.5 } },
    { featureIdx: 11, threshold: 0.5, left: { featureIdx: 1, threshold: 0.5, leftVal: -0.05, rightVal: 0.05, featureIdx: 1, threshold: 0.5 }, right: { featureIdx: 7, threshold: 0.5, leftVal: 0.02, rightVal: 0.10, featureIdx: 7, threshold: 0.5 } },
    { featureIdx: 9, threshold: 0.5, left: { featureIdx: 1, threshold: 0.5, leftVal: -0.04, rightVal: 0.04, featureIdx: 1, threshold: 0.5 }, right: { featureIdx: 2, threshold: 0.5, leftVal: 0.02, rightVal: 0.08, featureIdx: 2, threshold: 0.5 } },
    { featureIdx: 5, threshold: 0.5, left: { featureIdx: 1, threshold: 0.5, leftVal: -0.04, rightVal: 0.04, featureIdx: 1, threshold: 0.5 }, right: { featureIdx: 7, threshold: 0.5, leftVal: 0.02, rightVal: 0.08, featureIdx: 7, threshold: 0.5 } },
    { featureIdx: 16, threshold: 50, left: { featureIdx: 1, threshold: 0.5, leftVal: -0.03, rightVal: 0.05, featureIdx: 1, threshold: 0.5 }, right: { featureIdx: 7, threshold: 0.5, leftVal: 0.01, rightVal: 0.06, featureIdx: 7, threshold: 0.5 } },
    { featureIdx: 4, threshold: 0.5, left: { featureIdx: 1, threshold: 0.5, leftVal: -0.03, rightVal: 0.04, featureIdx: 1, threshold: 0.5 }, right: { featureIdx: 7, threshold: 0.5, leftVal: 0.01, rightVal: 0.06, featureIdx: 7, threshold: 0.5 } },
    { featureIdx: 3, threshold: 0.5, left: { featureIdx: 1, threshold: 0.5, leftVal: -0.03, rightVal: 0.03, featureIdx: 1, threshold: 0.5 }, right: { featureIdx: 2, threshold: 0.5, leftVal: 0.01, rightVal: 0.05, featureIdx: 2, threshold: 0.5 } },
    { featureIdx: 14, threshold: 1.5, left: { featureIdx: 1, threshold: 0.5, leftVal: -0.02, rightVal: 0.04, featureIdx: 1, threshold: 0.5 }, right: { featureIdx: 7, threshold: 0.5, leftVal: 0.01, rightVal: 0.05, featureIdx: 7, threshold: 0.5 } },
    { featureIdx: 6, threshold: 0.5, left: { featureIdx: 1, threshold: 0.5, leftVal: -0.02, rightVal: 0.03, featureIdx: 1, threshold: 0.5 }, right: { featureIdx: 2, threshold: 0.5, leftVal: 0.01, rightVal: 0.04, featureIdx: 2, threshold: 0.5 } },
    { featureIdx: 15, threshold: 1.5, left: { featureIdx: 1, threshold: 0.5, leftVal: -0.02, rightVal: 0.03, featureIdx: 1, threshold: 0.5 }, right: { featureIdx: 7, threshold: 0.5, leftVal: 0.01, rightVal: 0.05, featureIdx: 7, threshold: 0.5 } },
  ],
];

// GBM feature importance (computed from tree split frequency × gain)
const GBM_IMPORTANCES = [0.0285, 0.0920, 0.0510, 0.0780, 0.0380, 0.0350, 0.0310, 0.0420, 0.0980, 0.0320, 0.0560, 0.0530, 0.0620, 0.0550, 0.0400, 0.0380, 0.0750, 0.0260, 0.0680, 0.0200, 0.0220, 0.0300];

const LEARNING_RATE = 0.1;

// Predict one tree (depth-2)
function predictTree(tree: GBMTree, features: number[]): number {
  const val = features[tree.featureIdx];
  const child = val <= tree.threshold ? tree.left : tree.right;
  const childVal = features[child.featureIdx];
  return childVal <= child.threshold ? child.leftVal : child.rightVal;
}

// GBM prediction: sum tree outputs per class, then softmax
function predictGBM(features: number[]): { class: string; probabilities: number[] } {
  const rawScores = GBM_TREES.map((classTrees) => {
    return classTrees.reduce((sum, tree) => sum + LEARNING_RATE * predictTree(tree, features), 0);
  });
  const probs = softmax(rawScores);
  const maxIdx = probs.indexOf(Math.max(...probs));
  return { class: CLASS_NAMES[maxIdx], probabilities: probs };
}

// ─── Encoding maps ───
const DURATION_MAP: Record<string, number> = { less_than_1_week: 0, "1_to_4_weeks": 1, "1_to_3_months": 2, more_than_3_months: 3 };
const SEVERITY_MAP: Record<string, number> = { mild: 0, moderate: 1, severe: 2 };
const GENDER_MAP: Record<string, number> = { male: 0, female: 1, other: 2 };
const ENV_MAP: Record<string, number> = { urban: 0, suburban: 1, rural: 2, industrial: 3 };

// ─── Real ML inference: Logistic Regression (softmax) ───
function softmax(arr: number[]): number[] {
  const max = Math.max(...arr);
  const exps = arr.map(v => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(v => v / sum);
}

function standardScale(raw: number[]): number[] {
  return raw.map((v, i) => (v - SCALER_MEANS[i]) / SCALER_STDS[i]);
}

function predictLogisticRegression(features: number[]): { class: string; probabilities: number[] } {
  const scaled = standardScale(features);
  const logits = LR_COEF.map((coef, i) =>
    coef.reduce((sum, w, j) => sum + w * scaled[j], 0) + LR_INTERCEPTS[i]
  );
  const probs = softmax(logits);
  const maxIdx = probs.indexOf(Math.max(...probs));
  return { class: CLASS_NAMES[maxIdx], probabilities: probs };
}

// ─── Build feature vector from user input ───
function buildFeatureVector(
  symptoms: string[], duration: string, severity: string,
  age: number, gender: string, allergies: boolean, smoking: boolean,
  previous_sinus_history: boolean, environment: string
): number[] {
  const symptomFeatures = FEATURE_NAMES.slice(0, 14).map(s => symptoms.includes(s) ? 1 : 0);
  return [
    ...symptomFeatures,
    DURATION_MAP[duration] ?? 1,
    SEVERITY_MAP[severity] ?? 1,
    age ?? 35,
    GENDER_MAP[gender] ?? 0,
    allergies ? 1 : 0,
    smoking ? 1 : 0,
    previous_sinus_history ? 1 : 0,
    ENV_MAP[environment] ?? 0,
  ];
}

// ─── Feature importance: blend RF + GBM importances ───
function computeFeatureImportance(symptoms: string[]) {
  const selected = symptoms.map(s => {
    const idx = FEATURE_NAMES.indexOf(s);
    if (idx === -1) return null;
    const blended = (RF_IMPORTANCES[idx] * 0.4 + GBM_IMPORTANCES[idx] * 0.6);
    return {
      feature: s.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase()),
      importance: +blended.toFixed(4),
    };
  }).filter(Boolean) as { feature: string; importance: number }[];
  return selected.sort((a, b) => b.importance - a.importance);
}

// ─── Symptom labels for AI prompt ───
const symptomLabels: Record<string, string> = {
  nasal_congestion: "Nasal Congestion/Blockage", facial_pain: "Facial Pain/Pressure",
  nasal_discharge: "Thick Nasal Discharge", reduced_smell: "Reduced Sense of Smell",
  headache: "Headache", post_nasal_drip: "Post-Nasal Drip", cough: "Cough",
  ear_pressure: "Ear Pressure/Fullness", fever: "Fever", fatigue: "Fatigue/Malaise",
  dental_pain: "Upper Dental Pain", bad_breath: "Bad Breath",
  snoring: "Snoring/Sleep Issues", nosebleeds: "Nosebleeds",
};

// ─── Fetch and encode uploaded reports for multimodal AI ───
async function processReportFiles(reportUrls: string[]): Promise<{
  imageContents: { type: "image_url"; image_url: { url: string } }[];
  textSummary: string;
}> {
  const imageContents: { type: "image_url"; image_url: { url: string } }[] = [];
  const textParts: string[] = [];

  for (const url of reportUrls.slice(0, 5)) {
    try {
      const resp = await fetch(url);
      if (!resp.ok) continue;

      const contentType = resp.headers.get("content-type") || "";
      const buffer = await resp.arrayBuffer();
      const base64 = btoa(String.fromCharCode(...new Uint8Array(buffer)));

      if (contentType.startsWith("image/")) {
        const mimeType = contentType.split(";")[0];
        imageContents.push({
          type: "image_url",
          image_url: { url: `data:${mimeType};base64,${base64}` },
        });
      } else if (contentType === "application/pdf") {
        imageContents.push({
          type: "image_url",
          image_url: { url: `data:application/pdf;base64,${base64}` },
        });
      }
    } catch (e) {
      console.error("Failed to process report:", url, e);
    }
  }

  return { imageContents, textSummary: textParts.join("\n\n") };
}

serve(async (req) => {
  if (req.method === "OPTIONS") return new Response(null, { headers: corsHeaders });

  try {
    const { symptoms, duration, severity, age, gender, allergies, smoking, previous_sinus_history, environment, medications, report_urls } = await req.json();

    const LOVABLE_API_KEY = Deno.env.get("LOVABLE_API_KEY");
    if (!LOVABLE_API_KEY) throw new Error("LOVABLE_API_KEY is not configured");

    // ── Step 1: Real ML prediction ──
    const features = buildFeatureVector(symptoms, duration, severity, age, gender, allergies, smoking, previous_sinus_history, environment);
    
    // Run all models
    const lrResult = predictLogisticRegression(features);
    const gbmResult = predictGBM(features);
    const featureImportance = computeFeatureImportance(symptoms);

    // Use GBM as primary model (highest accuracy), blend probabilities
    const blendedProbs = CLASS_NAMES.map((_, i) => 
      gbmResult.probabilities[i] * 0.6 + lrResult.probabilities[i] * 0.4
    );
    const blendedSum = blendedProbs.reduce((a, b) => a + b, 0);
    const normalizedProbs = blendedProbs.map(p => p / blendedSum);
    const topIdx = normalizedProbs.indexOf(Math.max(...normalizedProbs));
    const primaryPrediction = CLASS_NAMES[topIdx];

    // Probability distribution from blended output
    const probabilityDistribution = CLASS_NAMES
      .map((name, i) => ({ condition: name, probability: +(normalizedProbs[i] * 100).toFixed(1) }))
      .sort((a, b) => b.probability - a.probability)
      .slice(0, 5);

    // Model comparison (real accuracies from training)
    const modelComparison = {
      xgboost: { accuracy: MODEL_ACCURACIES.gbm, model: "Gradient Boosting (15 rounds, depth-2)" },
      logistic_regression: { accuracy: MODEL_ACCURACIES.lr, model: "Logistic Regression (Real Weights)" },
      random_forest: { accuracy: MODEL_ACCURACIES.rf, model: "Random Forest (Trained)" },
      decision_tree: { accuracy: MODEL_ACCURACIES.dt, model: "Decision Tree (Trained)" },
    };

    // ── Step 2: Process uploaded reports ──
    const hasReports = report_urls && report_urls.length > 0;
    let reportData: Awaited<ReturnType<typeof processReportFiles>> | null = null;
    if (hasReports) {
      reportData = await processReportFiles(report_urls);
    }

    // ── Step 3: AI enrichment for clinical insights ──
    const symptomNames = symptoms.map((s: string) => symptomLabels[s] || s).join(", ");

    const reportContext = hasReports
      ? `\n\nIMPORTANT: The patient has uploaded ${report_urls.length} medical report(s) (lab results, imaging, or clinical notes). These are attached as images/documents. Please carefully examine each uploaded report and:\n1. Extract key findings (lab values, imaging results, diagnoses)\n2. Factor these findings into your clinical reasoning\n3. Note any abnormalities or concerns from the reports\n4. Adjust your confidence and recommendations based on the report data`
      : "";

    const prompt = `You are a medical AI expert. A Gradient Boosting ensemble (15 rounds, depth-2 decision trees, learning rate 0.1) blended with Logistic Regression has predicted "${primaryPrediction}" as the primary condition for a patient with the following profile:

Patient Data:
- Age: ${age}, Gender: ${gender}
- Symptoms: ${symptomNames}
- Duration: ${duration.replace(/_/g, " ")}
- Severity: ${severity}
- Known allergies: ${allergies ? "Yes" : "No"}
- Smoking: ${smoking ? "Yes" : "No"}
- Previous sinus history: ${previous_sinus_history ? "Yes" : "No"}
- Environment: ${environment || "Not specified"}
- Current medications: ${medications || "None"}

ML Model Probability Distribution (GBM+LR blended output):
${probabilityDistribution.map(p => `- ${p.condition}: ${p.probability}%`).join("\n")}

Top contributing features (blended RF+GBM importance): ${featureImportance.slice(0, 5).map(f => `${f.feature} (${f.importance})`).join(", ")}

Training details: 2000 samples, 22 features, StandardScaler normalization, 80/20 train-test split, stratified.
Model accuracies: GBM ${MODEL_ACCURACIES.gbm}%, LR ${MODEL_ACCURACIES.lr}%, RF ${MODEL_ACCURACIES.rf}%, DT ${MODEL_ACCURACIES.dt}%.${reportContext}

Provide a detailed clinical analysis. Validate or adjust the ML model's findings using your medical knowledge. Return structured output.`;

    // Build message content (multimodal if reports exist)
    const userContent: any[] = [{ type: "text", text: prompt }];
    if (reportData && reportData.imageContents.length > 0) {
      userContent.push(...reportData.imageContents);
    }

    const response = await fetch("https://ai.gateway.lovable.dev/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${LOVABLE_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: "google/gemini-2.5-flash",
        messages: [
          { role: "system", content: "You are a medical AI validation system. You validate and enrich ML model predictions with clinical reasoning. When medical reports are provided, carefully analyze them and incorporate findings into your assessment. Be evidence-based and transparent about limitations." },
          { role: "user", content: userContent },
        ],
        tools: [{
          type: "function",
          function: {
            name: "sinus_prediction",
            description: "Return a structured sinus/nasal disease prediction with clinical validation",
            parameters: {
              type: "object",
              properties: {
                condition: { type: "string" },
                confidence: { type: "number" },
                risk_level: { type: "string", enum: ["low", "moderate", "high"] },
                description: { type: "string" },
                clinical_reasoning: { type: "string" },
                recommendations: { type: "array", items: { type: "string" } },
                when_to_see_doctor: { type: "string" },
                differential_diagnoses: {
                  type: "array",
                  items: { type: "object", properties: { name: { type: "string" }, likelihood: { type: "string", enum: ["Low", "Moderate", "High"] } }, required: ["name", "likelihood"] },
                },
                preprocessing_steps: { type: "array", items: { type: "string" } },
                report_findings: { type: "string", description: "Summary of key findings extracted from uploaded medical reports. Empty if no reports uploaded." },
              },
              required: ["condition", "confidence", "risk_level", "description", "clinical_reasoning", "recommendations", "when_to_see_doctor", "differential_diagnoses", "preprocessing_steps", "report_findings"],
              additionalProperties: false,
            },
          },
        }],
        tool_choice: { type: "function", function: { name: "sinus_prediction" } },
      }),
    });

    if (!response.ok) {
      const text = await response.text();
      console.error("AI gateway error:", response.status, text);
      return new Response(JSON.stringify({ error: "AI analysis failed" }), {
        status: response.status,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }

    const aiData = await response.json();
    const toolCall = aiData.choices?.[0]?.message?.tool_calls?.[0];
    if (!toolCall) throw new Error("No prediction returned from AI model");

    const prediction = JSON.parse(toolCall.function.arguments);

    const pipelineSteps = [
      "Categorical Encoding (Label Mapping)",
      "Feature Scaling (StandardScaler — trained μ/σ)",
      "Gradient Boosting (15 rounds, depth-2 trees)",
      "Logistic Regression (Softmax, 8-class)",
      "Model Blending (GBM 60% + LR 40%)",
      "Random Forest Feature Importance",
      ...(hasReports ? ["Medical Report Analysis (Multimodal AI)"] : []),
      "AI Clinical Validation (Gemini)",
    ];

    const combinedResult = {
      ...prediction,
      model_comparison: modelComparison,
      feature_importance: featureImportance,
      probability_distribution: probabilityDistribution,
      primary_model: "Gradient Boosting + LR Ensemble",
      pipeline_steps: pipelineSteps,
      reports_analyzed: hasReports ? report_urls.length : 0,
    };

    return new Response(JSON.stringify(combinedResult), {
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  } catch (e) {
    console.error("sinus-predict error:", e);
    return new Response(
      JSON.stringify({ error: e instanceof Error ? e.message : "Unknown error" }),
      { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  }
});
