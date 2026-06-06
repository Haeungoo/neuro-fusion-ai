import {
  Activity,
  BarChart3,
  BrainCircuit,
  Database,
  FileText,
  Home,
  Menu,
  Settings,
  ShieldCheck,
  Users,
} from "lucide-react";
import Link from "next/link";

type AppShellProps = {
  children: React.ReactNode;
  activePath: "/" | "/mri" | "/seizure" | "/motor";
  title: string;
  subtitle: string;
};

const navigation = [
  { href: "/", label: "Overview", icon: Home },
  { href: "/mri", label: "MRI Tumor Segmentation", icon: BrainCircuit },
  { href: "/seizure", label: "EEG Seizure Detection", icon: Activity },
  { href: "/motor", label: "EEG Motor Imagery BCI", icon: BrainCircuit },
];

// const secondaryNavigation = [
//   { label: "Reports", icon: FileText },
//   { label: "Data Management", icon: Database },
//   { label: "Model Performance", icon: BarChart3 },
//   { label: "Settings", icon: Settings },
//   { label: "Users & Roles", icon: Users },
// ];

export default function AppShell({
  children,
  activePath,
  title,
  subtitle,
}: AppShellProps) {
  return (
    <main className="min-h-screen bg-slate-50 text-slate-900">
      <div className="grid min-h-screen lg:grid-cols-[230px_1fr]">
        <aside className="border-b border-slate-200 bg-white lg:border-b-0 lg:border-r">
          <div className="flex items-center gap-3 border-b border-slate-200 px-5 py-5">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-teal-50 text-teal-600">
              <BrainCircuit size={25} />
            </div>

            <div>
              <p className="font-bold text-teal-600">NeuroFusion-AI</p>
              <p className="text-xs text-slate-400">Clinical AI platform</p>
            </div>
          </div>

          <nav className="space-y-1 p-4">
            {navigation.map(({ href, label, icon: Icon }) => {
              const active = href === activePath;

              return (
                <Link
                  key={href}
                  href={href}
                  className={[
                    "flex items-center gap-3 rounded-xl px-3 py-3 text-sm",
                    "font-medium transition",
                    active
                      ? "bg-teal-600 text-white shadow-sm"
                      : "text-slate-600 hover:bg-slate-100 hover:text-slate-900",
                  ].join(" ")}
                >
                  <Icon size={19} strokeWidth={1.8} />
                  {label}
                </Link>
              );
            })}
          </nav>

          {/* <div className="mx-4 border-t border-slate-200" />

          <nav className="space-y-1 p-4">
            {secondaryNavigation.map(({ label, icon: Icon }) => (
              <button
                key={label}
                type="button"
                className="flex w-full items-center gap-3 rounded-xl px-3 py-3 text-left text-sm font-medium text-slate-600 hover:bg-slate-100"
              >
                <Icon size={18} strokeWidth={1.8} />
                {label}
              </button>
            ))}
          </nav> */}

          <section className="m-4 rounded-2xl border border-slate-200 bg-slate-50 p-4">
            <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
              System status
            </p>

            <div className="mt-4 space-y-3 text-xs text-slate-600">
              <StatusLine label="Backend (FastAPI)" />
              <StatusLine label="Frontend (Next.js)" />
              <StatusLine label="Prototype (Streamlit)" />
            </div>
          </section>
        </aside>

        <section className="min-w-0">
          <header className="flex min-h-20 items-center justify-between border-b border-slate-200 bg-white px-5 py-4 sm:px-8">
            <div className="flex items-center gap-4">
              <button
                type="button"
                className="rounded-lg p-2 text-slate-500 hover:bg-slate-100"
                aria-label="Open menu"
              >
                <Menu size={22} />
              </button>

              <div>
                <h1 className="text-xl font-bold text-slate-900">{title}</h1>
                <p className="mt-0.5 text-sm text-slate-500">{subtitle}</p>
              </div>
            </div>

            <div className="hidden items-center gap-5 md:flex">
              <div className="flex items-center gap-2 text-sm font-medium text-slate-600">
                <span className="h-2 w-2 rounded-full bg-emerald-500" />
                All systems operational
              </div>

              <div className="border-l border-slate-200 pl-5">
                <p className="text-sm font-semibold text-slate-800">
                  NeuroFusion-AI
                </p>
                <p className="text-xs text-slate-500">Research environment</p>
              </div>
            </div>
          </header>

          <div className="p-5 sm:p-7 lg:p-8">{children}</div>

          <footer className="border-t border-slate-200 bg-white px-6 py-4 text-center text-xs text-slate-500">
            Results are for research and educational use only and are not
            intended for clinical diagnosis.
          </footer>
        </section>
      </div>
    </main>
  );
}

function StatusLine({ label }: { label: string }) {
  return (
    <div className="flex items-center gap-2">
      <ShieldCheck size={15} className="text-emerald-500" />
      <span>{label}</span>
    </div>
  );
}