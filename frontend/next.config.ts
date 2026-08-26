import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Required for optimized Docker standalone deployment
  output: "standalone",
  // Expose API URL to client components
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1",
  },
};

export default nextConfig;
