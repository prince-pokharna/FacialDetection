import React, { useEffect, useState } from 'react';
import {
  Users,
  Calendar,
  Download,
  Trash2,
  Cpu,
  Search,
  CheckCircle,
  FileSpreadsheet,
  FileText,
  Activity,
  ShieldAlert,
} from 'lucide-react';
import { api, Person, AttendanceRecord, ModelInfo } from '../services/api';
import { useAuth } from '../context/AuthContext';

export const AdminDashboard: React.FC = () => {
  const { isAdmin } = useAuth();
  const [activeTab, setActiveTab] = useState<'attendance' | 'people' | 'model'>('attendance');

  // Data states
  const [people, setPeople] = useState<Person[]>([]);
  const [attendance, setAttendance] = useState<AttendanceRecord[]>([]);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [dateFilter, setDateFilter] = useState(new Date().toISOString().split('T')[0]);
  const [loading, setLoading] = useState(true);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const loadData = async () => {
    setLoading(true);
    setErrorMsg(null);
    try {
      const [peopleData, attendanceData, infoData] = await Promise.all([
        api.getPeople(),
        api.getAttendance(dateFilter || undefined),
        api.getModelInfo(),
      ]);
      setPeople(peopleData);
      setAttendance(attendanceData);
      setModelInfo(infoData);
    } catch (err: any) {
      setErrorMsg(err.message || 'Failed loading dashboard data');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, [dateFilter]);

  const handleDeletePerson = async (id: string, name: string) => {
    if (!window.confirm(`Permanently erase ${name}'s biometric embeddings and profile under Right to be Forgotten?`)) {
      return;
    }
    try {
      await api.deletePerson(id);
      setPeople((prev) => prev.filter((p) => p.id !== id));
      // Refresh model info to reflect gallery count update
      const info = await api.getModelInfo();
      setModelInfo(info);
    } catch (err: any) {
      alert(`Deletion failed: ${err.message}`);
    }
  };

  const filteredAttendance = attendance.filter((r) =>
    r.person_name.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="max-w-6xl mx-auto px-4 py-8">
      {/* Top Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 mb-8">
        <div>
          <h2 className="text-2xl font-bold text-slate-900 dark:text-white">
            Administration &amp; Analytics
          </h2>
          <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
            Biometric database records, CSV/Excel export, and system health metrics.
          </p>
        </div>

        {/* Tab Switcher */}
        <div className="flex items-center gap-1 p-1 bg-slate-200 dark:bg-slate-800 rounded-2xl">
          <button
            onClick={() => setActiveTab('attendance')}
            className={`px-4 py-2 rounded-xl text-xs font-bold transition ${
              activeTab === 'attendance'
                ? 'bg-white dark:bg-slate-900 text-sky-500 shadow-sm'
                : 'text-slate-600 dark:text-slate-400 hover:text-slate-900'
            }`}
          >
            Attendance Logs
          </button>
          <button
            onClick={() => setActiveTab('people')}
            className={`px-4 py-2 rounded-xl text-xs font-bold transition ${
              activeTab === 'people'
                ? 'bg-white dark:bg-slate-900 text-sky-500 shadow-sm'
                : 'text-slate-600 dark:text-slate-400 hover:text-slate-900'
            }`}
          >
            Manage People ({people.length})
          </button>
          <button
            onClick={() => setActiveTab('model')}
            className={`px-4 py-2 rounded-xl text-xs font-bold transition ${
              activeTab === 'model'
                ? 'bg-white dark:bg-slate-900 text-sky-500 shadow-sm'
                : 'text-slate-600 dark:text-slate-400 hover:text-slate-900'
            }`}
          >
            Model &amp; Thresholds
          </button>
        </div>
      </div>

      {/* Metrics Highlights Bar */}
      <div className="grid grid-cols-1 sm:grid-cols-4 gap-4 mb-8">
        <div className="p-5 rounded-3xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 shadow-sm">
          <div className="flex items-center justify-between text-slate-500 dark:text-slate-400 mb-2">
            <span className="text-xs font-bold uppercase tracking-wider">Enrolled People</span>
            <Users className="w-5 h-5 text-sky-500" />
          </div>
          <p className="text-3xl font-extrabold text-slate-900 dark:text-white">{people.length}</p>
        </div>

        <div className="p-5 rounded-3xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 shadow-sm">
          <div className="flex items-center justify-between text-slate-500 dark:text-slate-400 mb-2">
            <span className="text-xs font-bold uppercase tracking-wider">Today's Check-Ins</span>
            <CheckCircle className="w-5 h-5 text-emerald-500" />
          </div>
          <p className="text-3xl font-extrabold text-slate-900 dark:text-white">{attendance.length}</p>
        </div>

        <div className="p-5 rounded-3xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 shadow-sm">
          <div className="flex items-center justify-between text-slate-500 dark:text-slate-400 mb-2">
            <span className="text-xs font-bold uppercase tracking-wider">Attendance Rate</span>
            <Activity className="w-5 h-5 text-indigo-500" />
          </div>
          <p className="text-3xl font-extrabold text-slate-900 dark:text-white">
            {people.length > 0 ? `${((attendance.length / people.length) * 100).toFixed(0)}%` : '0%'}
          </p>
        </div>

        <div className="p-5 rounded-3xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 shadow-sm">
          <div className="flex items-center justify-between text-slate-500 dark:text-slate-400 mb-2">
            <span className="text-xs font-bold uppercase tracking-wider">Vector Gallery</span>
            <Cpu className="w-5 h-5 text-amber-500" />
          </div>
          <p className="text-3xl font-extrabold text-slate-900 dark:text-white">
            {modelInfo?.gallery_identities_loaded || people.length}
          </p>
        </div>
      </div>

      {/* Tab Content: Attendance Logs */}
      {activeTab === 'attendance' && (
        <div className="p-6 rounded-3xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 shadow-sm space-y-6">
          {/* Controls: Search, Date Filter, Export */}
          <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
            <div className="flex items-center gap-3 w-full sm:w-auto">
              <div className="relative flex-1 sm:w-64">
                <Search className="w-4 h-4 text-slate-400 absolute left-3 top-3" />
                <input
                  type="text"
                  placeholder="Search name..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="w-full pl-9 pr-4 py-2 rounded-xl bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-sm outline-none focus:ring-2 focus:ring-sky-500 text-slate-900 dark:text-white"
                />
              </div>

              <input
                type="date"
                value={dateFilter}
                onChange={(e) => setDateFilter(e.target.value)}
                className="px-3 py-2 rounded-xl bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-sm outline-none text-slate-900 dark:text-white"
              />
            </div>

            <div className="flex items-center gap-2">
              <button
                onClick={() => api.exportCsv(dateFilter)}
                className="px-3.5 py-2 rounded-xl bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-300 text-xs font-semibold flex items-center gap-1.5 transition"
              >
                <FileText className="w-4 h-4 text-sky-500" /> Export CSV
              </button>
              <button
                onClick={() => api.exportExcel(dateFilter)}
                className="px-3.5 py-2 rounded-xl bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-300 text-xs font-semibold flex items-center gap-1.5 transition"
              >
                <FileSpreadsheet className="w-4 h-4 text-emerald-500" /> Export Excel
              </button>
            </div>
          </div>

          {/* Table */}
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm text-slate-600 dark:text-slate-300">
              <thead className="text-xs uppercase bg-slate-50 dark:bg-slate-800/60 text-slate-500 dark:text-slate-400 border-b border-slate-200 dark:border-slate-800">
                <tr>
                  <th className="py-3 px-4">Person</th>
                  <th className="py-3 px-4">Confidence</th>
                  <th className="py-3 px-4">Liveness</th>
                  <th className="py-3 px-4">Session</th>
                  <th className="py-3 px-4">Time</th>
                  <th className="py-3 px-4">Method</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
                {filteredAttendance.length === 0 ? (
                  <tr>
                    <td colSpan={6} className="py-8 text-center text-slate-400">
                      No attendance records found for this date.
                    </td>
                  </tr>
                ) : (
                  filteredAttendance.map((rec) => (
                    <tr key={rec.id} className="hover:bg-slate-50/50 dark:hover:bg-slate-800/40">
                      <td className="py-3 px-4 font-semibold text-slate-900 dark:text-white">
                        {rec.person_name}
                      </td>
                      <td className="py-3 px-4">
                        <span className="font-mono text-xs bg-emerald-100 dark:bg-emerald-950/60 text-emerald-700 dark:text-emerald-400 px-2 py-0.5 rounded">
                          {(rec.confidence * 100).toFixed(1)}%
                        </span>
                      </td>
                      <td className="py-3 px-4">
                        <span className="text-xs font-medium text-emerald-600 dark:text-emerald-400 flex items-center gap-1">
                          <CheckCircle className="w-3.5 h-3.5" /> Live Skin
                        </span>
                      </td>
                      <td className="py-3 px-4">{rec.session_name}</td>
                      <td className="py-3 px-4 font-mono text-xs">
                        {new Date(rec.timestamp).toLocaleTimeString()}
                      </td>
                      <td className="py-3 px-4 text-xs font-mono text-slate-400">{rec.marked_by}</td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Tab Content: People Management */}
      {activeTab === 'people' && (
        <div className="p-6 rounded-3xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 shadow-sm space-y-4">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-bold text-slate-900 dark:text-white">
              Registered Identities &amp; Consent Records
            </h3>
            <span className="text-xs text-slate-500">
              Complies with Right to be Forgotten (biometric data erasure)
            </span>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm text-slate-600 dark:text-slate-300">
              <thead className="text-xs uppercase bg-slate-50 dark:bg-slate-800/60 text-slate-500 dark:text-slate-400 border-b border-slate-200 dark:border-slate-800">
                <tr>
                  <th className="py-3 px-4">Name</th>
                  <th className="py-3 px-4">ID</th>
                  <th className="py-3 px-4">Department</th>
                  <th className="py-3 px-4">Vectors Stored</th>
                  <th className="py-3 px-4">Consent Status</th>
                  <th className="py-3 px-4 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
                {people.map((p) => (
                  <tr key={p.id} className="hover:bg-slate-50/50 dark:hover:bg-slate-800/40">
                    <td className="py-3 px-4 font-semibold text-slate-900 dark:text-white">
                      {p.full_name}
                    </td>
                    <td className="py-3 px-4 font-mono text-xs">{p.external_id || '—'}</td>
                    <td className="py-3 px-4">{p.department || 'General'}</td>
                    <td className="py-3 px-4">
                      <span className="font-mono text-xs bg-sky-100 dark:bg-sky-950/60 text-sky-700 dark:text-sky-400 px-2 py-0.5 rounded">
                        {p.enrolled_embeddings_count} Prototype Vector(s)
                      </span>
                    </td>
                    <td className="py-3 px-4">
                      {p.consent_given ? (
                        <span className="text-xs font-semibold text-emerald-600 dark:text-emerald-400 flex items-center gap-1">
                          <CheckCircle className="w-3.5 h-3.5" /> Granted ({new Date(p.consent_timestamp || p.created_at).toLocaleDateString()})
                        </span>
                      ) : (
                        <span className="text-xs text-rose-500">No Consent</span>
                      )}
                    </td>
                    <td className="py-3 px-4 text-right">
                      {isAdmin && (
                        <button
                          onClick={() => handleDeletePerson(p.id, p.full_name)}
                          className="p-1.5 rounded-lg text-rose-500 hover:bg-rose-50 dark:hover:bg-rose-950/40 transition"
                          title="Erase Biometric Profile"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Tab Content: Model & Calibration Info */}
      {activeTab === 'model' && (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
          <div className="p-6 rounded-3xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 shadow-sm space-y-4">
            <h3 className="text-lg font-bold text-slate-900 dark:text-white flex items-center gap-2">
              <Cpu className="w-5 h-5 text-sky-500" />
              Biometric Architecture
            </h3>
            <div className="space-y-3 text-sm text-slate-600 dark:text-slate-300">
              <div className="flex justify-between py-1.5 border-b border-slate-100 dark:border-slate-800">
                <span className="text-slate-500">Paradigm</span>
                <span className="font-semibold text-slate-900 dark:text-white">{modelInfo?.architecture}</span>
              </div>
              <div className="flex justify-between py-1.5 border-b border-slate-100 dark:border-slate-800">
                <span className="text-slate-500">Detector</span>
                <span className="font-mono text-xs">{modelInfo?.detector}</span>
              </div>
              <div className="flex justify-between py-1.5 border-b border-slate-100 dark:border-slate-800">
                <span className="text-slate-500">Embedding Engine</span>
                <span className="font-semibold text-slate-900 dark:text-white">{modelInfo?.embedder}</span>
              </div>
              <div className="flex justify-between py-1.5 border-b border-slate-100 dark:border-slate-800">
                <span className="text-slate-500">Vector Dimension</span>
                <span className="font-mono text-xs font-bold">{modelInfo?.embedding_dimensions}-d (L2 Unit Vector)</span>
              </div>
              <div className="flex justify-between py-1.5 border-b border-slate-100 dark:border-slate-800">
                <span className="text-slate-500">Liveness PAD</span>
                <span className="font-mono text-xs">{modelInfo?.liveness_method}</span>
              </div>
            </div>
          </div>

          <div className="p-6 rounded-3xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 shadow-sm space-y-4">
            <h3 className="text-lg font-bold text-slate-900 dark:text-white flex items-center gap-2">
              <ShieldAlert className="w-5 h-5 text-emerald-500" />
              Calibrated Operating Thresholds
            </h3>
            <div className="space-y-3 text-sm text-slate-600 dark:text-slate-300">
              <div className="flex justify-between py-1.5 border-b border-slate-100 dark:border-slate-800">
                <span className="text-slate-500">Operating Point (FAR = 1.0%)</span>
                <span className="font-mono text-xs font-bold text-emerald-500">
                  Cosine Threshold &ge; {modelInfo?.thresholds.cosine_threshold}
                </span>
              </div>
              <div className="flex justify-between py-1.5 border-b border-slate-100 dark:border-slate-800">
                <span className="text-slate-500">Anti-Spoof Liveness Threshold</span>
                <span className="font-mono text-xs font-bold text-sky-500">
                  Score &ge; {modelInfo?.thresholds.liveness_threshold}
                </span>
              </div>
              <div className="flex justify-between py-1.5 border-b border-slate-100 dark:border-slate-800">
                <span className="text-slate-500">Smoothing Buffer Window</span>
                <span className="font-mono text-xs">{modelInfo?.thresholds.smoothing_window} frames</span>
              </div>
              <div className="flex justify-between py-1.5 border-b border-slate-100 dark:border-slate-800">
                <span className="text-slate-500">Per-Person Cooldown</span>
                <span className="font-mono text-xs">{modelInfo?.thresholds.cooldown_seconds} seconds</span>
              </div>
              <div className="flex justify-between py-1.5 border-b border-slate-100 dark:border-slate-800">
                <span className="text-slate-500">Loaded Gallery Vectors</span>
                <span className="font-mono text-xs font-bold">{modelInfo?.gallery_identities_loaded} identities</span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
