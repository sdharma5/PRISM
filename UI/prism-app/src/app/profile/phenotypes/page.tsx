// Consolidated into /overview.
//
// Overview is the single evidence profile. Redirect rather than delete so
// existing links and bookmarks still resolve.

import { redirect } from 'next/navigation'

export default function Page() {
  redirect('/overview')
}
