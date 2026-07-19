// Folded into /intake.
//
// Spec §16 describes one data-entry flow including the optional uploads, so it
// all lives at /intake. Redirect rather than delete so existing links resolve.

import { redirect } from 'next/navigation'

export default function Page() {
  redirect('/intake')
}
