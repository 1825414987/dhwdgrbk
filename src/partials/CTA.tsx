import {
  GradientText,
  Newsletter,
  Section,
} from 'astro-boilerplate-components';

const CTA = () => (
  <Section>
    <Newsletter
      title={
        <>
          本网站仅为个人记录 <GradientText>所用</GradientText>
        </>
      }
      description="有网站问题联系我（QQ）:***"
    />
  </Section>
);

export { CTA };
