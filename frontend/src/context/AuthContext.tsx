import React, { createContext, useContext, useState, useEffect } from 'react';
import { User, getAuthToken, setAuthToken, api } from '../services/api';

interface AuthContextType {
  user: User | null;
  token: string | null;
  login: (email: string, pass: string) => Promise<void>;
  logout: () => void;
  isAuthenticated: boolean;
  isAdmin: boolean;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [token, setTokenState] = useState<string | null>(getAuthToken());
  const [user, setUser] = useState<User | null>(() => {
    const saved = localStorage.getItem('attendance_user');
    return saved ? JSON.parse(saved) : null;
  });

  const login = async (email: string, pass: string) => {
    const res = await api.login(email, pass);
    const u: User = {
      id: res.user_id,
      email: res.email,
      full_name: res.full_name,
      role: res.role,
    };
    setAuthToken(res.access_token);
    setTokenState(res.access_token);
    setUser(u);
    localStorage.setItem('attendance_user', JSON.stringify(u));
  };

  const logout = () => {
    setAuthToken(null);
    setTokenState(null);
    setUser(null);
    localStorage.removeItem('attendance_user');
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        token,
        login,
        logout,
        isAuthenticated: !!token,
        isAdmin: user?.role === 'admin',
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) throw new Error('useAuth must be used within an AuthProvider');
  return context;
};
