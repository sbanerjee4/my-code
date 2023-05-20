import java.util.*;
import java.io.*;
import java.lang.*;

public class art
{
   private static int[] a;
   private static int m = 0;
   private static int n;
   public static void main(String[] args) throws FileNotFoundException
   {
      Scanner in = new Scanner(new File("art.in"));
      n = in.nextInt();
      a = new int[n];
      for(int i = 0; i < n; i++)
         a[i] = in.nextInt();
      in.close();
      
      boolean climb = false;
      boolean allUp = false, allDown = false;
      check(0);
      
      for(int i = 0; i < n - 1; i++)
         if(a[i + 1] > a[i]) climb = true;
         else if(a[i + 1] < a[i] && climb)
         {
            check(i);
            climb = false;
         }
      
      check(n - 1);
     
      for(int i = 0; i < n - 1; i++)
         if(a[i + 1] > a[i] && !allDown) allUp = true;
         else if(a[i + 1] < a[i] && !allUp) allDown = true;
         else if(a[i + 1] > a[i] && allDown)
         {
            allDown = false;
            break;
         }
         else if(a[i + 1] < a[i] && allUp)
         {
            allUp = false;
            break;
         }
      
      if(!(allUp || allDown)) System.out.println(m);
      else System.out.println(n);
   }
   
   public static void check(int pos)
   {
      int r = n - 1, l = 0;
      boolean rightBreak = false, leftBreak = false;
      for(int i = pos; i < n - 1; i++)
         if(a[i + 1] > a[i])
         {
            r = i;
            rightBreak = true;
            break;
         }
      
      for(int i = pos; i > 0; i--)
         if(a[i - 1] > a[i])
         {
            l = i;
            leftBreak = true;
            break;
         }
      
      if(pos == n - 1) r = n - 1;
      if(pos == 0) l = 0;
      
      int temp = 0;
      if(!rightBreak && !leftBreak) temp = n;
      else temp = Math.abs(r - l) + 1;
      m = Math.max(m, temp);
   }
}