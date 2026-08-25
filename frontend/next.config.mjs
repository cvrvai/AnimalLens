/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  images: {
    unoptimized: true,
  },
  async rewrites() {
    return [
      {
        source: '/api/backend/:path*',
        destination: 'http://127.0.0.1:8088/v1/:path*',
      },
      {
        source: '/static/:path*',
        destination: 'http://127.0.0.1:8088/static/:path*',
      },
    ];
  },
};

export default nextConfig;
