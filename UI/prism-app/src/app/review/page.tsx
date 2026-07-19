// Merged into /timeline.
//
// Review and Timeline called the identical three functions -- getEvents,
// confirmEvent, rejectEvent -- over the same data. Two pages, one dataset, and
// no way for a user to know which one to trust. Timeline is the survivor
// because it shows the full history; a confirmation queue is a filtered view of
// that, not a separate thing.
//
// Redirect rather than delete so existing links still resolve.

import { redirect } from 'next/navigation'

export default function Page() {
  redirect('/timeline')
}
