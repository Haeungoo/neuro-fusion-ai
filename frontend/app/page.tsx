import Link from "next/link";
import type { ElementType } from "react";
import {
  Activity,
  BrainCircuit,
  Database,
  Gauge,
  Layers3,
  ScanLine,
  ShieldCheck,
} from "lucide-react";

import AppShell from "@/components/AppShell";
import StatCard from "@/components/StatCard";
import StatusBadge from "@/components/StatusBadge";
import { mediaUrl } from "@/app/lib/api";

const modules: {
  title: string;
  description: string;
  href: string;
  icon: ElementType;
  iconClassName: string;
  model: string;
  dataset: string;
  metric: string;
  score: string;
  scoreClassName: string;
  previewSrc: string;
}[] = [
  {
    title: "MRI Tumor Segmentation",
    description:
      "Brain tumor segmentation from MRI scans using a 2D U-Net deep learning model.",
    href: "/mri",
    icon: BrainCircuit,
    iconClassName: "bg-violet-50 text-violet-600",
    model: "2D U-Net",
    dataset: "BraTS 2020",
    metric: "Dice Score",
    score: "0.87",
    scoreClassName: "text-violet-600",
    previewSrc: mediaUrl("mri/mri_prediction_overlay.png"),
  },
  {
    title: "EEG Seizure Detection",
    description:
      "Seizure vs. non-seizure classification from EEG signals using machine learning.",
    href: "/seizure",
    icon: Activity,
    iconClassName: "bg-blue-50 text-blue-600",
    model: "Random Forest",
    dataset: "CHB-MIT",
    metric: "F1 Score",
    score: "0.91",
    scoreClassName: "text-blue-600",
    previewSrc: mediaUrl("seizure/seizure_evaluation_confusion_matrix.png"),
  },
  {
    title: "EEG Motor Imagery BCI",
    description:
      "Left-hand versus right-hand motor imagery classification using CSP + LDA.",
    href: "/motor",
    icon: BrainCircuit,
    iconClassName: "bg-emerald-50 text-emerald-600",
    model: "CSP + LDA",
    dataset: "PhysioNet EEGBCI",
    metric: "Accuracy",
    score: "0.93",
    scoreClassName: "text-emerald-600",
    previewSrc: mediaUrl("motor_imagery/motor_imagery_confusion_matrix.png"),
  },
];

export default function OverviewPage() {
  return (
    <AppShell
      activePath="/"
      title="Overview"
      subtitle="Multimodal Neuroscience AI Dashboard"
    >
      <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-6">
        <StatCard
          icon={Layers3}
          label="AI Modules"
          value="3"
          description="Active modules"
          iconClassName="bg-teal-50 text-teal-600"
        />

        <StatCard
          icon={BrainCircuit}
          label="Models"
          value="3"
          description="Deployed models"
          iconClassName="bg-blue-50 text-blue-600"
        />

        <StatCard
          icon={Database}
          label="Datasets"
          value="3"
          description="Integrated datasets"
          iconClassName="bg-emerald-50 text-emerald-600"
        />

        <StatCard
          icon={Gauge}
          label="System Status"
          value="100%"
          description="All modules healthy"
          iconClassName="bg-cyan-50 text-cyan-600"
        />

        <StatCard
          icon={ScanLine}
          label="Last Run"
          value="2m ago"
          description="Recent analysis"
          iconClassName="bg-violet-50 text-violet-600"
        />

        <StatCard
          icon={ShieldCheck}
          label="API Status"
          value="Online"
          description="FastAPI connected"
          iconClassName="bg-amber-50 text-amber-600"
        />
      </section>

      <section className="mt-5 grid gap-5 xl:grid-cols-3">
        {modules.map((module) => (
          <ModuleCard
            key={module.title}
            title={module.title}
            description={module.description}
            href={module.href}
            icon={module.icon}
            iconClassName={module.iconClassName}
            model={module.model}
            dataset={module.dataset}
            metric={module.metric}
            score={module.score}
            scoreClassName={module.scoreClassName}
            previewSrc={module.previewSrc}
          />
        ))}
      </section>

      <section className="mt-5 grid gap-5 xl:grid-cols-3">
        <article className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm xl:col-span-2">
          <h2 className="text-lg font-semibold text-slate-900">
            System Architecture
          </h2>

          <p className="mt-1 text-sm text-slate-500">
            NeuroFusion-AI evolved from a prototype Streamlit app into a
            FastAPI + Next.js dashboard.
          </p>

          <div className="mt-5 grid gap-4 md:grid-cols-3">
            <ArchitectureCard
              title="v0.1 Streamlit"
              description="Rapid prototyping and visualization dashboard."
            />

            <ArchitectureCard
              title="v0.2 FastAPI"
              description="REST API backend serving results, metrics, and metadata."
            />

            <ArchitectureCard
              title="v0.3 Next.js"
              description="Production-style frontend dashboard for neuroscience AI modules."
            />
          </div>

          <div className="mt-5 grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
            {[
              "Python",
              "FastAPI",
              "Next.js",
              "Tailwind CSS",
              "Machine Learning",
            ].map((tool) => (
              <div
                key={tool}
                className="rounded-xl border border-slate-200 bg-white px-4 py-3 text-center text-sm font-semibold text-slate-700"
              >
                {tool}
              </div>
            ))}
          </div>
        </article>

        <article className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="flex items-start justify-between gap-4">
            <div>
              <h2 className="text-lg font-semibold text-slate-900">
                Recent Activity
              </h2>

              <p className="mt-1 text-sm text-slate-500">
                Latest module-level updates.
              </p>
            </div>

            <StatusBadge active />
          </div>

          <div className="mt-5 space-y-4">
            <ActivityRow
              label="MRI analysis completed"
              description="Validation metrics updated"
              tone="green"
            />

            <ActivityRow
              label="EEG seizure detection completed"
              description="CHB-MIT evaluation finished"
              tone="blue"
            />

            <ActivityRow
              label="Motor imagery analysis completed"
              description="PhysioNet subject search updated"
              tone="emerald"
            />

            <ActivityRow
              label="Dataset updated"
              description="Local results refreshed"
              tone="violet"
            />
          </div>
        </article>
      </section>

      <section className="mt-5 rounded-xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-800">
        Results are for research and educational use only and are not intended
        for clinical diagnosis.
      </section>
    </AppShell>
  );
}

function ModuleCard({
  title,
  description,
  href,
  icon: Icon,
  iconClassName,
  model,
  dataset,
  metric,
  score,
  scoreClassName,
  previewSrc,
}: {
  title: string;
  description: string;
  href: string;
  icon: ElementType;
  iconClassName: string;
  model: string;
  dataset: string;
  metric: string;
  score: string;
  scoreClassName: string;
  previewSrc: string;
}) {
  return (
    <article className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="flex items-start justify-between gap-4">
        <div className="flex items-start gap-4">
          <div
            className={[
              "flex h-12 w-12 items-center justify-center rounded-2xl",
              iconClassName,
            ].join(" ")}
          >
            <Icon size={28} />
          </div>

          <div>
            <h2 className="text-lg font-semibold text-slate-900">
              {title}
            </h2>

            <p className="mt-1 text-sm leading-6 text-slate-500">
              {description}
            </p>
          </div>
        </div>

        <StatusBadge active />
      </div>

      <div className="mt-5 flex h-56 items-center justify-center overflow-hidden rounded-xl border border-slate-200 bg-slate-950 p-3">
        <img
          src={previewSrc}
          alt={`${title} preview`}
          className="max-h-full max-w-full rounded-lg object-contain"
        />
      </div>

      <div className="mt-5 grid grid-cols-4 gap-3">
        <MiniInfo label="Model" value={model} />
        <MiniInfo label="Dataset" value={dataset} />
        <MiniInfo label="Metric" value={metric} />
        <MiniInfo
          label="Score"
          value={score}
          valueClassName={scoreClassName}
        />
      </div>

      <Link
        href={href}
        className="mt-5 flex h-11 items-center justify-center rounded-xl border border-teal-200 text-sm font-semibold text-teal-600 transition hover:bg-teal-50"
      >
        View Details →
      </Link>
    </article>
  );
}

function MiniInfo({
  label,
  value,
  valueClassName = "text-slate-900",
}: {
  label: string;
  value: string;
  valueClassName?: string;
}) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-3">
      <p className="text-xs text-slate-500">{label}</p>

      <p
        className={[
          "mt-1 truncate text-sm font-semibold",
          valueClassName,
        ].join(" ")}
      >
        {value}
      </p>
    </div>
  );
}

function ArchitectureCard({
  title,
  description,
}: {
  title: string;
  description: string;
}) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-5">
      <p className="text-base font-semibold text-slate-900">
        {title}
      </p>

      <p className="mt-2 text-sm leading-6 text-slate-500">
        {description}
      </p>
    </div>
  );
}

function ActivityRow({
  label,
  description,
  tone,
}: {
  label: string;
  description: string;
  tone: "green" | "blue" | "emerald" | "violet";
}) {
  const toneClass = {
    green: "bg-green-500",
    blue: "bg-blue-500",
    emerald: "bg-emerald-500",
    violet: "bg-violet-500",
  }[tone];

  return (
    <div className="flex items-start gap-3">
      <div
        className={[
          "mt-1 h-3 w-3 rounded-full",
          toneClass,
        ].join(" ")}
      />

      <div>
        <p className="text-sm font-semibold text-slate-900">
          {label}
        </p>

        <p className="text-xs text-slate-500">
          {description}
        </p>
      </div>
    </div>
  );
}