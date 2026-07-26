import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  turbopack: {
    // exclude backend from being watched/bundled
  },
  webpack: (config) => {
    config.watchOptions = {
      ignored: ["**/backend/**"],
    };
    return config;
  },
};

export default nextConfig;
