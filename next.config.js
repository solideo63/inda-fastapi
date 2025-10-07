/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: "https://inda-backend.hf.space/:path*", // backend URL
      },
    ];
  },
};

module.exports = nextConfig;