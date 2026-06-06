import { BrainCircuit, Database, Gauge, ScanLine } from "lucide-react";

import AppShell from "@/components/AppShell";
import StatCard from "@/components/StatCard";
import StatusBadge from "@/components/StatusBadge";
import { apiFetch, mediaUrl } from "@/lib/api";

type FileInfo = {
  path: string;
  exists: boolean;
};

type MriStatus = {
  module: string;
  dataset: string;
  model: string;
  model_file: FileInfo;
  outputs: {
    input_slice: FileInfo;
    ground_truth_mask: FileInfo;
    predicted_mask: FileInfo;
    overlay: FileInfo;
  };
  training_metrics: {
    best_val_loss?: number;
    epochs?: number;
    num_samples?: number;
    train_samples?: number;
    val_samples?: number;
  };
  inference_metrics: {
    sample_name?: string;
    dice_score?: number;
    threshold?: number;
    prediction_pixels?: number;
    ground_truth_pixels?: number;
  };
  disclaimer: string;
};

export default async function MriPage() {
  const status = await apiFetch<MriStatus>("/api/mri/status");

  const dice = status?.inference_metrics?.dice_score;
  const bestValLoss = status?.training_metrics?.best_val_loss;

  return (
    <AppShell
      activePath="/mri"
      title="MRI Tumor Segmentation"
      subtitle="BraTS MRI analysis using a 2D U-Net"
    >
      <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <StatCard
          icon={BrainCircuit}
          label="Model"
          value={status?.model ?? "2D U-Net"}
          description={status?.model_file.exists ? "Available" : "Missing"}
        />
        <StatCard
          icon={Database}
          label="Dataset"
          value={status?.dataset ?? "BraTS 2020"}
          description="FLAIR MRI"
          iconClassName="bg-blue-50 text-blue-600"
        />
        <StatCard
          icon={Gauge}
          label="Dice score"
          value={dice == null ? "N/A" : dice.toFixed(4)}
          description="Prediction overlap"
          iconClassName="bg-violet-50 text-violet-600"
        />
        <StatCard
          icon={ScanLine}
          label="Best val loss"
          value={bestValLoss == null ? "N/A" : bestValLoss.toFixed(4)}
          description="Training result"
          iconClassName="bg-emerald-50 text-emerald-600"
        />
      </section>

      <section className="mt-5 rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h2 className="text-lg font-semibold text-slate-900">
              Segmentation comparison
            </h2>
            <p className="mt-1 text-sm text-slate-500">
              Original MRI, annotation, model prediction, and final overlay.
            </p>
          </div>
          <StatusBadge active={status?.model_file.exists ?? false} />
        </div>

        <div className="mt-5 grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
          <ImagePanel
            title="Input MRI"
            src={mediaUrl("mri/mri_input_slice.png")}
          />
          <ImagePanel
            title="Ground truth"
            src={mediaUrl("mri/mri_ground_truth_mask.png")}
          />
          <ImagePanel
            title="Predicted mask"
            src={mediaUrl("mri/mri_predicted_mask.png")}
          />
          <ImagePanel
            title="Prediction overlay"
            src={mediaUrl("mri/mri_prediction_overlay.png")}
          />
        </div>
      </section>

      <section className="mt-5 grid gap-5 lg:grid-cols-2">
        <InfoPanel
          title="Inference details"
          values={[
            ["Sample", status?.inference_metrics?.sample_name ?? "N/A"],
            [
              "Threshold",
              String(status?.inference_metrics?.threshold ?? "N/A"),
            ],
            [
              "Predicted pixels",
              formatNumber(status?.inference_metrics?.prediction_pixels),
            ],
            [
              "Ground-truth pixels",
              formatNumber(status?.inference_metrics?.ground_truth_pixels),
            ],
          ]}
        />

        <InfoPanel
          title="Training details"
          values={[
            [
              "Epochs",
              String(status?.training_metrics?.epochs ?? "N/A"),
            ],
            [
              "Total slices",
              formatNumber(status?.training_metrics?.num_samples),
            ],
            [
              "Training slices",
              formatNumber(status?.training_metrics?.train_samples),
            ],
            [
              "Validation slices",
              formatNumber(status?.training_metrics?.val_samples),
            ],
          ]}
        />
      </section>

      <section className="mt-5 rounded-xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-800">
        {status?.disclaimer ??
          "This model is for research and educational use only."}
      </section>
    </AppShell>
  );
}

function ImagePanel({ title, src }: { title: string; src: string }) {
  return (
    <figure className="overflow-hidden rounded-xl border border-slate-200">
      <div className="flex h-56 items-center justify-center bg-slate-950 p-3">
        <img src={src} alt={title} className="max-h-full max-w-full object-contain" />
      </div>
      <figcaption className="bg-white px-4 py-3 text-center text-sm font-medium text-slate-700">
        {title}
      </figcaption>
    </figure>
  );
}

function InfoPanel({
  title,
  values,
}: {
  title: string;
  values: Array<[string, string]>;
}) {
  return (
    <article className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
      <h2 className="text-lg font-semibold text-slate-900">{title}</h2>

      <dl className="mt-4 divide-y divide-slate-100">
        {values.map(([label, value]) => (
          <div
            key={label}
            className="flex items-center justify-between gap-5 py-3"
          >
            <dt className="text-sm text-slate-500">{label}</dt>
            <dd className="truncate text-sm font-semibold text-slate-800">
              {value}
            </dd>
          </div>
        ))}
      </dl>
    </article>
  );
}

function formatNumber(value?: number): string {
  return value == null ? "N/A" : value.toLocaleString();
}