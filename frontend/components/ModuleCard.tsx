import type { LucideIcon } from "lucide-react";
import Link from "next/link";

import StatusBadge from "./StatusBadge";

type ModuleCardProps = {
  title: string;
  description: string;
  icon: LucideIcon;
  iconClassName: string;
  imageSrc?: string;
  imageAlt: string;
  model: string;
  dataset: string;
  metric: string;
  score: string;
  href: string;
  buttonClassName: string;
  children?: React.ReactNode;
};

export default function ModuleCard({
  title,
  description,
  icon: Icon,
  iconClassName,
  imageSrc,
  imageAlt,
  model,
  dataset,
  metric,
  score,
  href,
  buttonClassName,
  children,
}: ModuleCardProps) {
  return (
    <article className="flex h-full flex-col rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="flex items-start justify-between gap-4">
        <div className="flex gap-3">
          <div
            className={`flex h-11 w-11 shrink-0 items-center justify-center rounded-xl ${iconClassName}`}
          >
            <Icon size={23} strokeWidth={1.8} />
          </div>

          <div>
            <h2 className="text-lg font-semibold text-slate-900">{title}</h2>
            <p className="mt-1 text-sm leading-5 text-slate-500">
              {description}
            </p>
          </div>
        </div>

        <StatusBadge />
      </div>

      <div className="mt-5 flex min-h-52 items-center justify-center overflow-hidden rounded-xl border border-slate-200 bg-slate-950">
        {children ??
          (imageSrc ? (
            <img
              src={imageSrc}
              alt={imageAlt}
              className="h-full max-h-64 w-full object-contain"
            />
          ) : (
            <p className="text-sm text-slate-400">Preview unavailable</p>
          ))}
      </div>

      <div className="mt-4 grid grid-cols-2 gap-2 sm:grid-cols-4">
        <Metric label="Model" value={model} />
        <Metric label="Dataset" value={dataset} />
        <Metric label="Metric" value={metric} />
        <Metric label="Score" value={score} emphasis />
      </div>

      <Link
        href={href}
        className={`mt-4 block rounded-xl border px-4 py-3 text-center text-sm font-semibold transition ${buttonClassName}`}
      >
        View details →
      </Link>
    </article>
  );
}

function Metric({
  label,
  value,
  emphasis = false,
}: {
  label: string;
  value: string;
  emphasis?: boolean;
}) {
  return (
    <div className="rounded-xl border border-slate-200 bg-slate-50 p-3">
      <p className="text-xs font-medium text-slate-500">{label}</p>
      <p
        className={[
          "mt-1 truncate text-sm font-semibold",
          emphasis ? "text-teal-700" : "text-slate-900",
        ].join(" ")}
        title={value}
      >
        {value}
      </p>
    </div>
  );
}