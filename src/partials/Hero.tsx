import {
  GradientText,
  HeroAvatar,
  HeroSocial,
  Section,
} from 'astro-boilerplate-components';

const Hero = () => (
  <Section>
    <HeroAvatar
      title={
        <>
          大家好，我是<GradientText>本博客创建者邓浩伟</GradientText> 👋
        </>
      }
      description={
        <>
          在这里我们分享编程技巧，项目经验以及文学分享、峡谷秘闻......{' '}
          <a
            className="text-cyan-400 hover:underline"
            href="https://xinghuo.xfyun.cn/desk"
          >
            本网站
          </a>{' '}
          专用于写开源的一些{' '}
          <a className="text-cyan-400 hover:underline" href="/">
            项目文档
          </a>{' '}
          并且分享一些生活中的碎片，分享是一件快乐的事...
        </>
      }
      avatar={
        <img
          className="w-50 h-80"
          src="/assets/images/hsxy.svg"
          alt="Avatar image"
          loading="lazy"
        />
      }
      socialButtons={
        <>
          <a href="/">
            <HeroSocial
              src="/assets/images/twitter-icon.png"
              alt="Twitter icon"
            />
          </a>
          <a href="/">
            <HeroSocial
              src="/assets/images/facebook-icon.png"
              alt="Facebook icon"
            />
          </a>
          <a href="/">
            <HeroSocial
              src="/assets/images/linkedin-icon.png"
              alt="Linkedin icon"
            />
          </a>
          <a href="/">
            <HeroSocial
              src="/assets/images/youtube-icon.png"
              alt="Youtube icon"
            />
          </a>
        </>
      }
    />
  </Section>
);

export { Hero };
