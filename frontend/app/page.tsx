import {
  Activity,
  BrainCircuit,
  Clock3,
  Database,
  Layers3,
  ShieldCheck,
} from "lucide-react";

import AppShell from "@/components/AppShell";
import ModuleCard from "@/components/ModuleCard";
import StatCard from "@/components/StatCard";
import { apiFetch, mediaUrl } from "@/lib/api";

type ModuleInfo = {
  task: string;
  model: string;
  dataset: string;
};

type OverviewResponse = {
  project: string;
  description: string;
  version: string;
  modules: {
    mri: ModuleInfo;
    seizure: ModuleInfo;
    motor: ModuleInfo;
  };
};

type MriStatus = {
  model_file: { exists: boolean };
  inference_metrics: {
    dice_score?: number;
  };
};

type MotorStatus = {
  model_file: { exists: boolean };
  metrics: {
    accuracy?: number;
    f1_score?: number;
  };
};

export default async function HomePage() {
  const [overview, mriStatus, motorStatus] = await Promise.all([
    apiFetch<OverviewResponse>("/api/overview"),
    apiFetch<MriStatus>("/api/mri/status"),
    apiFetch<MotorStatus>("/api/motor/status"),
  ]);

  const mriScore = mriStatus?.inference_metrics?.dice_score;
  const motorScore = motorStatus?.metrics?.accuracy;

  return (
    <AppShell
      activePath="/"
      title="Overview"
      subtitle="Multimodal Neuroscience AI Dashboard"
    >
      <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-5">
        <StatCard
          icon={Layers3}
          label="AI Modules"
          value="3"
          description="Active"
        />
        <StatCard
          icon={BrainCircuit}
          label="Models"
          value="3"
          description="Configured"
          iconClassName="bg-blue-50 text-blue-600"
        />
        <StatCard
          icon={Database}
          label="Datasets"
          value="3"
          description="Integrated"
          iconClassName="bg-emerald-50 text-emerald-600"
        />
        <StatCard
          icon={Clock3}
          label="Last run"
          value="Recent"
          description="Local results"
          iconClassName="bg-violet-50 text-violet-600"
        />
        <StatCard
          icon={ShieldCheck}
          label="System status"
          value="100%"
          description="Operational"
        />
      </section>

      <section className="mt-5 grid gap-5 xl:grid-cols-3">
        <ModuleCard
          title="MRI Tumor Segmentation"
          description={
            overview?.modules.mri.task ??
            "Brain tumor segmentation from MRI scans."
          }
          icon={BrainCircuit}
          iconClassName="bg-violet-50 text-violet-600"
          imageSrc={mediaUrl("mri/mri_prediction_overlay.png")}
          imageAlt="MRI tumor segmentation overlay"
          model={overview?.modules.mri.model ?? "2D U-Net"}
          dataset={overview?.modules.mri.dataset ?? "BraTS 2020"}
          metric="Dice score"
          score={mriScore == null ? "N/A" : mriScore.toFixed(3)}
          href="/mri"
          buttonClassName="border-violet-200 text-violet-700 hover:bg-violet-50"
        />

        <ModuleCard
          title="EEG Seizure Detection"
          description={
            overview?.modules.seizure.task ??
            "Seizure and non-seizure classification from EEG signals."
          }
          icon={Activity}
          iconClassName="bg-blue-50 text-blue-600"
          imageSrc={mediaUrl(
            "seizure/chbmit_multi_file_probability_timeline.png",
          )}
          imageAlt="EEG seizure probability timeline"
          model={overview?.modules.seizure.model ?? "Random Forest"}
          dataset={overview?.modules.seizure.dataset ?? "CHB-MIT"}
          metric="F1 score"
          score="See details"
          href="/seizure"
          buttonClassName="border-blue-200 text-blue-700 hover:bg-blue-50"
        />

        <ModuleCard
          title="EEG Motor Imagery BCI"
          description={
            overview?.modules.motor.task ??
            "Motor imagery classification for BCI applications."
          }
          icon={BrainCircuit}
          iconClassName="bg-emerald-50 text-emerald-600"
          imageSrc={mediaUrl("motor/csp_patterns.png")}
          imageAlt="Motor imagery CSP spatial patterns"
          model={overview?.modules.motor.model ?? "CSP + LDA"}
          dataset={overview?.modules.motor.dataset ?? "PhysioNet EEGBCI"}
          metric="Accuracy"
          score={motorScore == null ? "N/A" : motorScore.toFixed(3)}
          href="/motor"
          buttonClassName="border-emerald-200 text-emerald-700 hover:bg-emerald-50"
        />
      </section>

      <section className="mt-5 grid gap-5 xl:grid-cols-[1.3fr_0.9fr]">
        <ArchitecturePanel />

        <article className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <h2 className="text-lg font-semibold text-slate-900">
            Platform summary
          </h2>

          <p className="mt-3 text-sm leading-6 text-slate-600">
            {overview?.description ??
              "Connect the FastAPI backend to display project metadata."}
          </p>

          <div className="mt-5 space-y-3">
            <ActivityRow
              title="MRI analysis pipeline"
              subtitle={
                mriStatus?.model_file.exists
                  ? "Trained model detected"
                  : "Model file unavailable"
              }
            />
            <ActivityRow
              title="Motor imagery pipeline"
              subtitle={
                motorStatus?.model_file.exists
                  ? "CSP + LDA model detected"
                  : "Model file unavailable"
              }
            />
            <ActivityRow
              title="Next.js frontend"
              subtitle="FastAPI-connected medical UI"
            />
          </div>
        </article>
      </section>
    </AppShell>
  );
}

function ArchitecturePanel() {
  return (
    <article className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
      <h2 className="text-lg font-semibold text-slate-900">
        System Architecture
      </h2>

      <div className="mt-5 grid gap-4 md:grid-cols-3">
        <ArchitectureItem
          version="v0.1"
          title="Streamlit"
          description="Rapid prototype and model visualization."
        />
        <ArchitectureItem
          version="v0.2"
          title="FastAPI"
          description="REST backend serving status, metrics, and results."
        />
        <ArchitectureItem
          version="v0.3"
          title="Next.js"
          description="Production-style frontend for research workflows."
        />
      </div>
    </article>
  );
}

function ArchitectureItem({
  version,
  title,
  description,
}: {
  version: string;
  title: string;
  description: string;
}) {
  return (
    <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
      <p className="text-xs font-semibold text-teal-600">{version}</p>
      <p className="mt-1 font-semibold text-slate-900">{title}</p>
      <p className="mt-2 text-xs leading-5 text-slate-500">{description}</p>
    </div>
  );
}

function ActivityRow({
  title,
  subtitle,
}: {
  title: string;
  subtitle: string;
}) {
  return (
    <div className="flex items-start gap-3 rounded-xl bg-slate-50 p-3">
      <span className="mt-1 h-2 w-2 rounded-full bg-emerald-500" />
      <div>
        <p className="text-sm font-medium text-slate-800">{title}</p>
        <p className="mt-0.5 text-xs text-slate-500">{subtitle}</p>
      </div>
    </div>
  );
}