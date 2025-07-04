// import { BackgroundCircleProvider } from "@/components/main_page";
//
// export default function Home() {
//   return (
//     <div className="flex flex-col items-center justify-center h-screen">
//       <BackgroundCircleProvider />
//     </div>
//   );
// }

// import { UnifiedAvatarChat } from "@/components/UnifiedAvatarChat";
//
// export default function Home() {
//   return (
//     <div className="w-full h-screen">
//       <UnifiedAvatarChat />
//     </div>
//   );
// }

import { EnhancedVRMAvatarChat } from "@/components/UnifiedAvatarChat";

export default function Home() {
  return (
    <div className="w-full h-screen">
      <EnhancedVRMAvatarChat />
    </div>
  );
}