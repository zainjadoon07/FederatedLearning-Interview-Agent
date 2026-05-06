"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { useAuthStore } from "@/store/authStore";

export default function AuthGuard({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const { isAuthenticated, token } = useAuthStore();
  const [checked, setChecked] = useState(false);

  useEffect(() => {
    // Wait for Zustand to rehydrate from localStorage before deciding
    const hasToken = !!token || !!localStorage.getItem("token");
    if (!hasToken) {
      router.replace("/login");
    } else {
      setChecked(true);
    }
  }, [token, router]);

  if (!checked) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-[#0f0f13]">
        <div className="w-8 h-8 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  return <>{children}</>;
}
