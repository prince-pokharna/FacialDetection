/**
 * API client for Face Recognition Attendance System.
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || '';

export interface User {
  id: string;
  email: string;
  full_name: string;
  role: 'admin' | 'instructor';
}

export interface Person {
  id: string;
  external_id?: string;
  full_name: string;
  department?: string;
  consent_given: boolean;
  consent_timestamp?: string;
  created_at: string;
  enrolled_embeddings_count: number;
}

export interface AttendanceRecord {
  id: string;
  person_id: string;
  person_name: string;
  confidence: number;
  liveness_score: number;
  session_name: string;
  date: string;
  timestamp: string;
  marked_by: string;
}

export interface ModelInfo {
  architecture: string;
  detector: string;
  embedder: string;
  embedding_dimensions: number;
  liveness_method: string;
  thresholds: {
    cosine_threshold: number;
    liveness_threshold: number;
    smoothing_window: number;
    cooldown_seconds: number;
  };
  gallery_identities_loaded: number;
}

export function getAuthToken(): string | null {
  return localStorage.getItem('attendance_token');
}

export function setAuthToken(token: string | null) {
  if (token) {
    localStorage.setItem('attendance_token', token);
  } else {
    localStorage.removeItem('attendance_token');
  }
}

async function request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
  const token = getAuthToken();
  const headers = new Headers(options.headers || {});
  if (token && !headers.has('Authorization')) {
    headers.set('Authorization', `Bearer ${token}`);
  }

  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    ...options,
    headers,
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(errorData.detail?.message || errorData.detail || 'Request failed');
  }

  if (response.status === 204) {
    return {} as T;
  }

  return response.json();
}

export const api = {
  // Auth
  async login(email: string, password: string) {
    return request<{ access_token: string; user_id: string; email: string; role: 'admin' | 'instructor'; full_name: string }>(
      '/auth/login',
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password }),
      }
    );
  },

  async register(email: string, password: string, full_name: string, role = 'instructor') {
    return request<User>('/auth/register', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password, full_name, role }),
    });
  },

  // People
  async getPeople(search?: string) {
    const q = search ? `?search=${encodeURIComponent(search)}` : '';
    return request<Person[]>(`/people${q}`);
  },

  async createPerson(data: { full_name: string; external_id?: string; department?: string; consent_given: boolean }) {
    return request<Person>('/people', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
  },

  async deletePerson(personId: string) {
    return request<void>(`/people/${personId}`, {
      method: 'DELETE',
    });
  },

  async enrollFaces(personId: string, imageBlobs: Blob[]) {
    const formData = new FormData();
    imageBlobs.forEach((blob, idx) => {
      formData.append('images', blob, `face_${idx}.jpg`);
    });

    const token = getAuthToken();
    const headers: Record<string, string> = {};
    if (token) headers['Authorization'] = `Bearer ${token}`;

    const res = await fetch(`${API_BASE_URL}/people/${personId}/enroll`, {
      method: 'POST',
      headers,
      body: formData,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Enrollment failed' }));
      throw new Error(err.detail?.message || err.detail || 'Enrollment failed');
    }
    return res.json();
  },

  // Recognition
  async recognizeFrame(imageBase64: string, sessionName = 'Main', autoMark = true) {
    return request<{
      faces_detected: number;
      faces: Array<{
        bbox: [number, number, number, number];
        detection_score: number;
        is_live: boolean;
        liveness_score: number;
        spoof_type: string | null;
        is_known: boolean;
        person_id: string | null;
        person_name: string | null;
        confidence: number;
      }>;
      attendance_marked: boolean;
      marked_person_name: string | null;
      message: string;
    }>('/recognize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image_base64: imageBase64, session_name: sessionName, auto_mark: autoMark }),
    });
  },

  // Attendance
  async getAttendance(date?: string, personId?: string) {
    const params = new URLSearchParams();
    if (date) params.set('target_date', date);
    if (personId) params.set('person_id', personId);
    return request<AttendanceRecord[]>(`/attendance?${params.toString()}`);
  },

  async exportCsv(date?: string) {
    const token = getAuthToken();
    const url = `${API_BASE_URL}/attendance/export.csv${date ? `?target_date=${date}` : ''}`;
    const res = await fetch(url, {
      headers: token ? { Authorization: `Bearer ${token}` } : {},
    });
    const blob = await res.blob();
    const downloadUrl = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = downloadUrl;
    a.download = `attendance_${date || 'all'}.csv`;
    a.click();
  },

  async exportExcel(date?: string) {
    const token = getAuthToken();
    const url = `${API_BASE_URL}/attendance/export.xlsx${date ? `?target_date=${date}` : ''}`;
    const res = await fetch(url, {
      headers: token ? { Authorization: `Bearer ${token}` } : {},
    });
    const blob = await res.blob();
    const downloadUrl = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = downloadUrl;
    a.download = `attendance_${date || 'all'}.xlsx`;
    a.click();
  },

  // System
  async getHealth() {
    return request<{ status: string; app_name: string; database_connected: boolean; enrolled_gallery_size: number }>('/health');
  },

  async getModelInfo() {
    return request<ModelInfo>('/model/info');
  },
};
