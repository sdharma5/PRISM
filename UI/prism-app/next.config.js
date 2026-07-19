/** @type {import('next').NextConfig} */

// This checkout is shared on a cluster filesystem, and more than one person may
// run `next dev` against it at once. Next writes build output to a single
// `.next` and prunes files it does not recognise, so concurrent dev servers need
// separate build directories.
//
// `distDir` is per-user by default -- `npm run dev` gives `.next-alice` and
// `.next-bob` with no flag to remember. Set NEXT_DIST_DIR to override.
const user = process.env.USER || process.env.USERNAME || 'shared'

const nextConfig = {
  distDir: process.env.NEXT_DIST_DIR || `.next-${user}`,
  experimental: {
    serverComponentsExternalPackages: [],
  },
}

module.exports = nextConfig
